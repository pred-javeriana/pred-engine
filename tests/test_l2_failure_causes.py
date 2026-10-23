"""Una unidad fallida explica su causa en los tres archivos de la corrida."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pred_engine.comun.dataclasses.modelos_clasicos import ResultadoSeleccionClasica
from pred_engine.optimizacion.optimizadores.HPO.espacio import EspacioBusqueda, Flotante
from pred_engine.optimizacion.optimizadores.HPO.estudio import ejecutar_estudio
from pred_engine.optimizacion.optimizadores.modelos_clasicos import estrategia
from pred_engine.pipeline import PipelineInput
from pred_engine.run_artifacts import write_run
from pred_engine.verify import representative_csv, verification_pipeline
from tests._dobles import fabrica_rota


def limite_nivel(configuracion):
    return configuracion["nivel"] < 0


@pytest.mark.parametrize("invalido", [True, False], ids=["restriccion", "ajuste"])
def test_causa_l2_en_unidades_corrida_y_seleccion(tmp_path, monkeypatch, invalido):
    source = representative_csv(tmp_path / "source.csv")
    panel = pd.read_csv(source)
    pd.concat([panel, panel.assign(sku_id="broken")]).to_csv(source, index=False)
    original = estrategia.seleccionar_configuracion_clasica

    def seleccionar(serie, *, sku_id, **kwargs):
        if sku_id != "broken":
            return original(serie, sku_id=sku_id, **kwargs)
        estudio = ejecutar_estudio(
            serie,
            EspacioBusqueda(
                parametros=(Flotante("nivel", 1, 2),),
                restricciones=(limite_nivel,) if invalido else (),
            ),
            fabrica_rota,
            n_trials=1,
            min_train=16,
            horizonte=2,
            paso=3,
            familia="classical",
            sku_id=sku_id,
            raiz_corrida=tmp_path / "run" / "hpo",
            run_id="classical-broken-s",
        )
        return ResultadoSeleccionClasica(None, estudio)

    monkeypatch.setattr(estrategia, "seleccionar_configuracion_clasica", seleccionar)
    result = verification_pipeline().run(
        PipelineInput(csv_path=source, data_root=tmp_path / "data")
    )
    assert result.blocked_stage == "L4"
    assert len(result.failures) == 1
    error = result.failures[0].record.error
    causa = "configuracion_invalida" if invalido else "ajuste_fallido"
    detalle = "limite_nivel" if invalido else "RuntimeError"
    assert causa in error
    assert detalle in error
    archivos = write_run(result, tmp_path / "run", run_id="s", metadata={})
    unidades = [
        json.loads(line) for line in archivos["unidades"].read_text().splitlines()
    ]
    fallida = next(u for u in unidades if u["sku_id"] == "broken")
    assert fallida["error"] == error
    corrida = json.loads(archivos["corrida"].read_text())
    assert corrida["summary"]["failures"][0]["error"] == error
    seleccion = [
        json.loads(line) for line in archivos["seleccion"].read_text().splitlines()
    ]
    broken = next(s for s in seleccion if s["sku_id"] == "broken")
    familia = next(f for f in broken["families"] if f["family"] == "classical")
    assert causa in familia["error"]["text"]
    assert detalle in familia["error"]["text"]
    assert familia["trials"][0]["motivo"].startswith(
        "configuracion_invalida:" if invalido else "ventana=1 ajuste_fallido:"
    )
