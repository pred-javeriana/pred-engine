"""Productor M2 del manifiesto (3.2-A1): todos los candidatos, desde forecast_config."""

from __future__ import annotations

import json
from typing import Any

import pytest
from tests._manifiestos import CONFIGURACIONES, CONTEXTO, EMITIDO_EN

from pred_engine.comun.modelos.manifiesto_candidatos import (
    VERSION_ESQUEMA_MANIFIESTO,
    ManifiestoCandidatos,
)
from pred_engine.comun.modelos.modelos_fundacionales.configuracion import (
    CHRONOS2_ZERO_SHOT,
)
from pred_engine.optimizacion.router import (
    SelectionContractError,
    SelectionResult,
    construir_manifiesto,
)


def _resultado(familia: str, **cambios: Any) -> SelectionResult:
    datos: dict[str, Any] = {
        "sku_id": "S1",
        "sku_class": "smooth",
        "family": familia,
        "profile": "dense_stable",
        "produced_by": "Prueba",
        "payload": {"evidencia": "no viaja"},
        "forecast_config": {}
        if familia == "foundation"
        else dict(CONFIGURACIONES[familia]),
        "forecast_seed": 11,
    }
    return SelectionResult(**(datos | cambios))


def _construir(*resultados: SelectionResult) -> dict[str, Any]:
    contenido = construir_manifiesto(
        resultados, run_id_m2="m2-1", contexto=CONTEXTO, emitido_en=EMITIDO_EN
    )
    ManifiestoCandidatos.model_validate_json(contenido)
    return json.loads(contenido)


def test_incluye_todos_los_candidatos_sin_elegir_campeon() -> None:
    datos = _construir(
        *(_resultado(f) for f in ("classical", "ml", "dl", "foundation"))
    )
    assert datos["schema_version"] == VERSION_ESQUEMA_MANIFIESTO
    assert datos["run_id_m2"] == "m2-1"
    assert [c["candidato_id"] for c in datos["candidatos"]] == [
        "S1/classical/sarima",
        "S1/ml/lightgbm",
        "S1/dl/mlp",
        "S1/foundation/chronos2",
    ]


def test_usa_forecast_config_y_forecast_seed_no_el_payload() -> None:
    (entrada,) = _construir(_resultado("ml"))["candidatos"]
    assert entrada["configuracion"] == CONFIGURACIONES["ml"]
    assert entrada["semilla"] == 11
    assert "evidencia" not in json.dumps(entrada)


def test_la_familia_fundacional_declara_su_configuracion_congelada() -> None:
    (entrada,) = _construir(_resultado("foundation"))["candidatos"]
    assert entrada["configuracion"] == CHRONOS2_ZERO_SHOT.descripcion_canonica()


def test_resultado_sin_forecast_config_se_rechaza() -> None:
    with pytest.raises(SelectionContractError, match="forecast_config"):
        _construir(_resultado("ml", forecast_config=None))


def test_lista_vacia_produce_un_manifiesto_sin_candidatos() -> None:
    assert _construir()["candidatos"] == []
