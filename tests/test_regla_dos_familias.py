"""Regla de dos familias al cerrar L2 (ADR-023).

Un SKU con dos o mas familias enrutadas y una sola evaluable sale de la
competencia con su causa: no llega a L3 ni recibe veredicto en M3.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pred_engine.comun.errores_codigo import ELEGIBILIDAD_MENOS_DE_DOS_FAMILIAS
from pred_engine.comun.exclusiones import SkuExclusion
from pred_engine.forecasting.control_reanudacion import ControladorEvaluacion
from pred_engine.forecasting.persistencia_evidencia import aplicar_migraciones_m3
from pred_engine.optimizacion.optimizadores.modelos_clasicos import (
    estrategia as estrategia_clasica,
)
from pred_engine.optimizacion.optimizadores.modelos_machine_learning import (
    estrategia as estrategia_ml,
)
from pred_engine.pipeline import (
    EvaluationSettings,
    PipelineExecutionError,
    PipelineInput,
    summarize_run,
)
from pred_engine.pipeline_setup import build_pipeline
from tests._plataforma import crear_base_plataforma
from tests._series import serie_ar_estacional

pytestmark = pytest.mark.slow

DEMANDA = np.round(serie_ar_estacional(120), 2)
AJUSTES = EvaluationSettings(40, horizon=2, step=14, seasonality=7, metric="mae")
# Con esta semilla el HPO de SARIMA completa un ensayo para el SKU sano.
SEMILLA = 15
CAUSA = (
    "menos de dos familias evaluables tras L2: evaluable ml; "
    "fallidas classical (selection.no_viable_trial)"
)


def _entrada(raiz: Path, *skus: str) -> PipelineInput:
    fuente = raiz / "ventas.csv"
    n = len(DEMANDA)
    pd.concat(
        pd.DataFrame(
            {
                "sku_id": [sku] * n,
                "timestamp": pd.date_range("2024-01-01", periods=n),
                "demand_qty": DEMANDA + i,
                "lead_time_days": [3] * n,
            }
        )
        for i, sku in enumerate(skus)
    ).to_csv(fuente, index=False)
    return PipelineInput(csv_path=fuente, data_root=raiz / "data")


def _sin_ensayos(monkeypatch, modulo, nombre: str, rotos: set[str]) -> None:
    """El HPO de la familia no completa ningun ensayo para los SKU de ``rotos``."""
    original = getattr(modulo, nombre)

    def seleccionar(serie, *, sku_id, **kwargs):
        if sku_id in rotos:
            return SimpleNamespace(
                seleccionada=None,
                estudio=SimpleNamespace(resumen_causas="ajuste_fallido=1"),
            )
        return original(serie, sku_id=sku_id, **kwargs)

    monkeypatch.setattr(modulo, nombre, seleccionar)


def test_un_sku_con_una_sola_familia_evaluable_sale_antes_de_l3(tmp_path, monkeypatch):
    _sin_ensayos(
        monkeypatch,
        estrategia_clasica,
        "seleccionar_configuracion_clasica",
        {"solo-ml"},
    )
    pipeline = build_pipeline(
        AJUSTES, families=("classical", "ml"), n_trials=1, seed=SEMILLA
    )
    pipeline.progress = tmp_path / "progreso.jsonl"
    resultado = pipeline.run(_entrada(tmp_path, "sano", "solo-ml"))

    fitting, evaluation = resultado.fitting, resultado.evaluation
    assert fitting.exclusions == (
        SkuExclusion("solo-ml", CAUSA, ELEGIBILIDAD_MENOS_DE_DOS_FAMILIAS),
    )
    # L2 conserva las dos unidades del SKU; ni su candidato ni L3 siguen.
    assert sorted((u.sku_id, u.family, u.record.estado) for u in fitting.units) == [
        ("sano", "classical", "completada"),
        ("sano", "ml", "completada"),
        ("solo-ml", "classical", "fallida"),
        ("solo-ml", "ml", "completada"),
    ]
    assert {c.selection.sku_id for c in fitting.candidates} == {"sano"}
    assert {u.sku_id for u in evaluation.units} == {"sano"}
    assert fitting.m3_skus == {"sano"}
    resumen = summarize_run(resultado)
    assert resumen["exclusions"] == [
        {
            "sku_id": "solo-ml",
            "cause": CAUSA,
            "code": ELEGIBILIDAD_MENOS_DE_DOS_FAMILIAS,
        }
    ]
    eventos = [
        json.loads(linea)
        for linea in pipeline.progress.read_text().splitlines()
        if json.loads(linea)["event"] == "excluded"
    ]
    assert [(e["sku_id"], e["cause"], e["code"]) for e in eventos] == [
        ("solo-ml", CAUSA, ELEGIBILIDAD_MENOS_DE_DOS_FAMILIAS)
    ]


@pytest.mark.parametrize(
    ("familias", "rotas"),
    [
        # Sin familias evaluables el SKU sigue a M3, que lo juzga FALLO_TECNICO.
        (("classical", "ml"), ("classical", "ml")),
        # Con una sola familia enrutada no hay competencia que proteger.
        (("classical",), ("classical",)),
    ],
    ids=["ninguna-evaluable", "una-enrutada"],
)
def test_la_regla_no_excluye_sin_familias_ni_con_una_sola_enrutada(
    tmp_path, monkeypatch, familias, rotas
):
    if "classical" in rotas:
        _sin_ensayos(
            monkeypatch,
            estrategia_clasica,
            "seleccionar_configuracion_clasica",
            {"roto"},
        )
    if "ml" in rotas:
        _sin_ensayos(
            monkeypatch, estrategia_ml, "seleccionar_configuracion_ml", {"roto"}
        )
    pipeline = build_pipeline(AJUSTES, families=familias, n_trials=1, seed=SEMILLA)
    resultado = pipeline.run(_entrada(tmp_path, "sano", "roto"))

    assert resultado.fitting.exclusions == ()
    assert {c.selection.sku_id for c in resultado.fitting.candidates} == {"sano"}
    assert resultado.fitting.m3_skus == {"sano", "roto"}


def test_m3_no_juzga_al_sku_excluido_tras_l2(tmp_path, monkeypatch):
    _sin_ensayos(
        monkeypatch,
        estrategia_clasica,
        "seleccionar_configuracion_clasica",
        {"solo-ml"},
    )
    ruta = crear_base_plataforma(tmp_path / "central.db")
    conn = sqlite3.connect(ruta)
    aplicar_migraciones_m3(conn)
    pipeline = build_pipeline(
        AJUSTES, families=("classical", "ml"), n_trials=1, seed=SEMILLA
    )
    pipeline.validator = ControladorEvaluacion(
        conn,
        tmp_path / "corrida",
        run_id_m2="m2-dos-familias",
        codigo_version="revision-1",
        datos_sinteticos=True,
        data_root=tmp_path / "data",
    )
    resultado = pipeline.run(_entrada(tmp_path, "sano", "solo-ml"))

    assert resultado.complete is True
    assert [v.sku_id for v in resultado.validation.verdicts] == ["sano"]
    assert conn.execute("SELECT sku FROM m3_skus").fetchall() == [("sano",)]
    (procedencia,) = conn.execute("SELECT procedencia FROM m3_protocolos").fetchone()
    procedencia = json.loads(procedencia)
    assert procedencia["exclusiones_m2"] == [
        {
            "sku_id": "solo-ml",
            "cause": CAUSA,
            "code": ELEGIBILIDAD_MENOS_DE_DOS_FAMILIAS,
        }
    ]
    # La unidad clasica fallida sigue en la procedencia de L2.
    assert {(f["sku"], f["familia"]) for f in procedencia["fallos_m2"]} == {
        ("solo-ml", "classical")
    }
    # La publicacion lista la exclusion con su categoria, para el embudo de M4.
    assert conn.execute(
        "SELECT sku, sku_class, codigo, causa FROM m3_exclusiones"
    ).fetchall() == [("solo-ml", "smooth", ELEGIBILIDAD_MENOS_DE_DOS_FAMILIAS, CAUSA)]
    conn.close()


def test_sin_skus_que_juzgar_m3_falla_antes_de_registrar_la_corrida(
    tmp_path, monkeypatch
):
    _sin_ensayos(
        monkeypatch,
        estrategia_clasica,
        "seleccionar_configuracion_clasica",
        {"solo-ml"},
    )
    conn = sqlite3.connect(crear_base_plataforma(tmp_path / "central.db"))
    aplicar_migraciones_m3(conn)
    pipeline = build_pipeline(
        AJUSTES, families=("classical", "ml"), n_trials=1, seed=SEMILLA
    )
    pipeline.validator = ControladorEvaluacion(
        conn,
        tmp_path / "corrida",
        run_id_m2="m2-sin-skus",
        codigo_version="revision-1",
        datos_sinteticos=True,
        data_root=tmp_path / "data",
    )
    with pytest.raises(
        PipelineExecutionError,
        match=r"ningun SKU llega a M3: 1 SKU excluidos antes de M3 "
        r"\(eligibility\.fewer_than_two_families\)",
    ):
        pipeline.run(_entrada(tmp_path, "solo-ml"))
    assert conn.execute("SELECT COUNT(*) FROM m3_corridas").fetchone() == (0,)
    assert conn.execute("SELECT COUNT(*) FROM m3_protocolos").fetchone() == (0,)
    conn.close()
