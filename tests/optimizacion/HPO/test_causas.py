"""Las causas del HPO llegan al error publico de cada familia."""

from __future__ import annotations

from datetime import datetime, timedelta
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from tests._dobles import fabrica_rota

from pred_engine.comun.ensayos_hpo import leer_ensayos
from pred_engine.comun.modelos import ClassifiedObservation
from pred_engine.optimizacion.optimizadores.HPO.contratos import InfoTrial
from pred_engine.optimizacion.optimizadores.HPO.errores import EstudioError
from pred_engine.optimizacion.optimizadores.HPO.espacio import EspacioBusqueda, Flotante
from pred_engine.optimizacion.optimizadores.HPO.estudio import ejecutar_estudio
from pred_engine.optimizacion.optimizadores.HPO.registro import (
    instantanea_desde_estudio,
)
from pred_engine.optimizacion.router import SelectionContractError, SelectionRequest


def limite_nivel(configuracion):
    return configuracion["nivel"] < 0


def test_solo_sorteos_invalidos_explica_la_restriccion_y_persiste(tmp_path):
    with pytest.raises(EstudioError) as error:
        ejecutar_estudio(
            np.arange(30, dtype=float),
            EspacioBusqueda(
                parametros=(Flotante("nivel", 1, 2),),
                restricciones=(limite_nivel,),
            ),
            fabrica_rota,
            n_trials=1,
            min_train=20,
            raiz_corrida=tmp_path,
            run_id="invalido",
        )
    assert "configuracion_invalida" in str(error.value)
    assert "limite_nivel" in str(error.value)
    ensayos = leer_ensayos(tmp_path / "invalido")
    assert len(ensayos) == 21
    assert {t.motivo for t in ensayos} == {
        "configuracion_invalida:HPO.test_causas.limite_nivel"
    }


def test_fallo_de_ajuste_conserva_contador_y_causa():
    resultado = ejecutar_estudio(
        np.arange(30, dtype=float),
        EspacioBusqueda(parametros=(Flotante("nivel", 1, 2),)),
        fabrica_rota,
        n_trials=2,
        min_train=20,
    )
    assert resultado.causas == {"ajuste_fallido": 2}
    assert resultado.n_fallidos == 2
    assert resultado.n_podados == 0
    assert "ajuste_fallido" in resultado.resumen_causas
    assert "RuntimeError" in resultado.resumen_causas


@pytest.mark.parametrize(
    ("familia", "clase", "selector"),
    [
        ("clasicos", "ClassicalSelectionStrategy", "seleccionar_configuracion_clasica"),
        ("machine_learning", "MLSelectionStrategy", "seleccionar_configuracion_ml"),
        ("deep_learning", "DLSelectionStrategy", "seleccionar_configuracion_dl"),
    ],
)
def test_cada_estrategia_explica_el_fallo_de_ajuste(
    monkeypatch, familia, clase, selector
):
    estudio = ejecutar_estudio(
        np.arange(30, dtype=float),
        EspacioBusqueda(parametros=(Flotante("nivel", 1, 2),)),
        fabrica_rota,
        n_trials=2,
        min_train=20,
    )
    modulo = import_module(
        f"pred_engine.optimizacion.optimizadores.modelos_{familia}.estrategia"
    )
    resultado = (
        estudio
        if familia == "deep_learning"
        else SimpleNamespace(estudio=estudio, seleccionada=None)
    )
    monkeypatch.setattr(modulo, selector, lambda *args, **kwargs: resultado)
    request = SelectionRequest(
        sku_id="S1",
        sku_class="smooth",
        series=tuple(
            ClassifiedObservation(
                sku_id="S1",
                sku_class="smooth",
                timestamp=datetime(2024, 1, 1) + timedelta(days=i),
                demand_qty=float(i + 1),
                lead_time_days=1,
            )
            for i in range(30)
        ),
    )
    with pytest.raises(SelectionContractError) as error:
        getattr(modulo, clase)().select(request, "dense_stable")
    assert "causas={'ajuste_fallido': 2}" in str(error.value)
    assert "ventana=1 ajuste_fallido: RuntimeError" in str(error.value)


def test_causas_agrupa_prefijos_sin_perder_motivos():
    motivos = [
        "configuracion_invalida:limite_nivel",
        "ventana=1 ajuste_fallido: RuntimeError: sin convergencia",
        "ventana=2 valor=1 poda_semantica:prediccion_nula",
        "escalon=4 valor=3 umbral=1",
        "escalon=12 valor=4 umbral=2",
    ]
    resultado = instantanea_desde_estudio(
        [
            InfoTrial(
                numero=i,
                estado="fallido" if i < 2 else "podado",
                valor=None if i < 2 else 1.0,
                parametros={},
                atributos={"motivo": motivo},
            )
            for i, motivo in enumerate(motivos)
        ],
        ventanas=(),
        metrica_objetivo="mae",
        seed=17,
    )
    assert resultado.causas == {
        "configuracion_invalida": 1,
        "ajuste_fallido": 1,
        "poda_semantica": 1,
        "escalon": 2,
    }
    assert resultado.n_fallidos == 2
    assert resultado.n_podados == 3
    assert "sin convergencia" in resultado.resumen_causas
    assert "prediccion_nula" in resultado.resumen_causas
