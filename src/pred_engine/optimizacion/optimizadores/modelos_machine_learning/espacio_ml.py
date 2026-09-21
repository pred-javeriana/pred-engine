"""Espacio de busqueda de hiperparametros de ML (LightGBM) para el HPO.

Es la UNICA pieza propia de la familia ML: el resto de la seleccion (TPE,
ASHA, Walk-Forward, reanudacion) lo aporta `HPO/estudio.py`. Reemplaza al
papel que p,d,q juegan en los modelos clasicos: aqui la complejidad la fijan
hiperparametros de capacidad (`max_depth`, `n_estimators`, `lags`) y de
regularizacion (`min_child_weight`, `reg_*`, `subsample`, `colsample`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pred_engine.comun.modelos.modelos_machine_learning.lgbm import MUESTRAS_MINIMAS
from pred_engine.optimizacion.optimizadores.HPO.espacio import (
    Entero,
    EspacioBusqueda,
    Flotante,
)


@dataclass(frozen=True, slots=True)
class EspacioML:
    lags_min: int = 3
    lags_max: int = 14
    n_estimators_min: int = 20
    n_estimators_max: int = 300
    max_depth_min: int = 2
    max_depth_max: int = 8
    learning_rate_min: float = 0.01
    learning_rate_max: float = 0.3
    min_child_weight_min: float = 1.0
    min_child_weight_max: float = 20.0
    subsample_min: float = 0.5
    colsample_min: float = 0.5
    reg_min: float = 1e-8
    reg_max: float = 10.0
    # Cota de costo de entrenamiento: n_estimators * max_depth. Sin ella TPE
    # gasta el presupuesto en las esquinas mas caras del espacio.
    costo_max: int = 1500
    m: int = 7


def construir_espacio_ml(espacio: EspacioML | None = None) -> EspacioBusqueda:
    cfg = espacio or EspacioML()

    def _costo_acotado(configuracion: Mapping[str, Any]) -> bool:
        return configuracion["n_estimators"] * configuracion["max_depth"] <= (
            cfg.costo_max
        )

    return EspacioBusqueda(
        parametros=(
            Entero("lags", cfg.lags_min, cfg.lags_max),
            Entero("n_estimators", cfg.n_estimators_min, cfg.n_estimators_max),
            Entero("max_depth", cfg.max_depth_min, cfg.max_depth_max),
            Flotante(
                "learning_rate", cfg.learning_rate_min, cfg.learning_rate_max, log=True
            ),
            Flotante(
                "min_child_weight",
                cfg.min_child_weight_min,
                cfg.min_child_weight_max,
                log=True,
            ),
            Flotante("subsample", cfg.subsample_min, 1.0),
            Flotante("colsample_bytree", cfg.colsample_min, 1.0),
            Flotante("reg_alpha", cfg.reg_min, cfg.reg_max, log=True),
            Flotante("reg_lambda", cfg.reg_min, cfg.reg_max, log=True),
        ),
        restricciones=(_costo_acotado,),
    )


def min_train_recomendado_ml(espacio: EspacioML | None = None) -> int:
    """Menor `min_train` con el que TODA configuracion del espacio puede ajustarse."""
    cfg = espacio or EspacioML()
    base = 2 * cfg.m if cfg.m > 1 else 0
    return cfg.lags_max + max(base, 2 * MUESTRAS_MINIMAS)
