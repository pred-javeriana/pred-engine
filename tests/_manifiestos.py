"""Constructores de manifiestos M2 -> M3 validos para las pruebas de 3.2.

No empieza con `test_`: pytest no lo recolecta; lo importan los `test_*.py`.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import Any

from pred_engine.comun.modelos.manifiesto_candidatos import (
    MODELO_POR_FAMILIA,
    VERSION_ESQUEMA_MANIFIESTO,
    ContextoParticion,
)
from pred_engine.comun.modelos.modelos_fundacionales.configuracion import (
    CHRONOS2_ZERO_SHOT,
)

CONTEXTO = ContextoParticion(
    ingesta_ref_m1="a" * 64,
    t_corte_reserva=date(2024, 3, 1),
    fraccion_reserva=0.2,
)
EMITIDO_EN = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)

# Configuraciones completas y pequenas (ajustan en milisegundos).
CONFIGURACIONES: dict[str, dict[str, Any]] = {
    "classical": {"p": 1, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "m": 7},
    "ml": {
        "lags": 3,
        "m": 7,
        "n_estimators": 5,
        "max_depth": 2,
        "learning_rate": 0.1,
        "min_child_weight": 1.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    },
    "dl": {
        "lags": 3,
        "capas": 2,
        "unidades": 4,
        "epochs": 2,
        "batch_size": 8,
        "learning_rate": 0.01,
        "dropout": 0.0,
        "l2": 1e-4,
    },
    "foundation": CHRONOS2_ZERO_SHOT.descripcion_canonica(),
}


def candidato(familia: str = "ml", sku: str = "S1", **cambios: Any) -> dict[str, Any]:
    modelo = MODELO_POR_FAMILIA[familia]
    base = {
        "candidato_id": f"{sku}/{familia}/{modelo}",
        "sku": sku,
        "sku_class": "smooth",
        "familia": familia,
        "modelo": modelo,
        "semilla": 7,
        "configuracion": dict(CONFIGURACIONES[familia]),
    }
    return base | cambios


def con_configuracion(familia: str, **cambios: Any) -> dict[str, Any]:
    return candidato(familia, configuracion=CONFIGURACIONES[familia] | cambios)


def sin_campo(datos: dict[str, Any], campo: str) -> dict[str, Any]:
    return {clave: valor for clave, valor in datos.items() if clave != campo}


def manifiesto(*candidatos: dict[str, Any], **cambios: Any) -> str:
    datos = {
        "schema_version": VERSION_ESQUEMA_MANIFIESTO,
        "run_id_m2": "m2-corrida-1",
        "emitido_en": EMITIDO_EN.isoformat(),
        "contexto": json.loads(CONTEXTO.model_dump_json()),
        "candidatos": list(candidatos),
    }
    return json.dumps(datos | cambios)
