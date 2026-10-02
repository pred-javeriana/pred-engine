"""Manifiesto M2 -> M3 (ADR-03-004): ida y vuelta, y sin defaults implicitos."""

from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic import ValidationError

from pred_engine.optimizacion.manifiesto_candidatos import (
    ContextoCorte,
    ManifiestoCandidatos,
    candidato_desde_seleccion,
    construir_manifiesto,
)
from pred_engine.optimizacion.router import SelectionResult

_EVIDENCIA = {
    "metrica_objetivo": "mase",
    "valor": 0.82,
    "n_ventanas": 40,
    "n_trials": 25,
    "n_completados": 20,
    "n_podados": 5,
    "n_fallidos": 0,
}
_CONTEXTO = ContextoCorte(
    t_estrella=date(2025, 11, 4),
    primer_dia_reservado=date(2025, 11, 5),
    ultimo_dia_observado=date(2026, 2, 12),
    dias_reservados=100,
    fraccion_reservada=0.2,
)


def _seleccion(familia: str, configuracion: dict, **payload) -> SelectionResult:
    return SelectionResult(
        sku_id="100::syn001",
        sku_class="intermittent",
        family=familia,  # type: ignore[arg-type]
        profile="sparse_stable",
        produced_by="prueba",
        policy_version="2.2.0-initial",
        forecast_config=configuracion,
        forecast_seed=7,
        payload=payload,
    )


_SARIMA = {"p": 1, "d": 0, "q": 1, "P": 1, "D": 0, "Q": 0, "m": 7, "tendencia": "c"}
_LIGHTGBM = {
    "lags": 7,
    "m": 7,
    "n_estimators": 120,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_weight": 2.0,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.001,
    "reg_lambda": 0.01,
}


def _manifiesto() -> ManifiestoCandidatos:
    return construir_manifiesto(
        [
            _seleccion("classical", _SARIMA, estudio_hpo="classical-x", **_EVIDENCIA),
            _seleccion("ml", _LIGHTGBM, estudio_hpo="ml-x", **_EVIDENCIA),
            _seleccion("foundation", {}, optimizado=False),
        ],
        run_id="r-1",
        contexto=_CONTEXTO,
        versiones={"pred-engine": "0.1.0"},
    )


def test_el_manifiesto_sobrevive_la_ida_y_vuelta_por_json():
    manifiesto = _manifiesto()
    leido = ManifiestoCandidatos.model_validate_json(manifiesto.model_dump_json())

    assert leido == manifiesto
    clasico, ml, fundacional = leido.candidatos
    assert (clasico.familia, clasico.modelo) == ("classical", "sarima")
    assert clasico.configuracion.tendencia == "c"
    assert clasico.corrida.estudio_hpo == "classical-x"
    assert ml.configuracion.n_estimators == 120
    assert fundacional.modelo == "chronos-2"
    assert fundacional.estadistico_puntual == "mediana"
    assert fundacional.configuracion.revision
    assert fundacional.evidencia_hpo is None
    assert leido.contexto.t_estrella == date(2025, 11, 4)


@pytest.mark.parametrize(
    ("familia", "configuracion"),
    [
        ("classical", {k: v for k, v in _SARIMA.items() if k != "tendencia"}),
        ("ml", {k: v for k, v in _LIGHTGBM.items() if k != "reg_lambda"}),
        ("classical", {**_SARIMA, "trend": "c"}),
        ("ml", {**_LIGHTGBM, "subsample": 1.5}),
    ],
)
def test_rechaza_configuraciones_incompletas_extra_o_fuera_de_rango(
    familia, configuracion
):
    with pytest.raises(ValidationError):
        candidato_desde_seleccion(
            _seleccion(familia, configuracion, **_EVIDENCIA), run_id="r-1"
        )


def test_rechaza_candidatos_repetidos_o_de_otra_corrida():
    datos = _manifiesto().model_dump(mode="json")
    repetido = {**datos, "candidatos": datos["candidatos"][:1] * 2}
    with pytest.raises(ValidationError, match="repetido"):
        ManifiestoCandidatos.model_validate_json(json.dumps(repetido))
    ajeno = {**datos, "run_id": "r-2"}
    with pytest.raises(ValidationError, match="run_id"):
        ManifiestoCandidatos.model_validate_json(json.dumps(ajeno))
