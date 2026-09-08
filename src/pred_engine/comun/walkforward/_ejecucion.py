"""Ejecucion de UNA ventana: la unica pieza que tocan fit/predict.

Compartida bit a bit por el modo integro (`walk_forward.py`) y el modo
greedy (`walk_forward_greedy.py`). Es lo que garantiza la equivalencia
exacta entre ambos modos: si cada uno reimplementara su propio "ajustar y
medir", podrian divergir sutilmente y producir metricas no comparables.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from pred_engine.comun.dataclasses.validacion_temporal import (
    ResultadoVentana,
    VentanaTemporal,
)
from pred_engine.comun.walkforward.metricas import calcular_metricas
from pred_engine.comun.walkforward.protocolos import FabricaPronosticador
from pred_engine.comun.walkforward.semillas import derivar_semilla

_METRICAS_FALLO = {
    "mae": float("inf"),
    "rmse": float("inf"),
    "smape": float("inf"),
    "mase": float("inf"),
}


def ejecutar_ventana(
    y: np.ndarray,
    fabrica: FabricaPronosticador,
    configuracion: Mapping[str, Any],
    ventana: VentanaTemporal,
    *,
    estacionalidad: int,
    seed: int,
    identificador: str,
    tolerar_fallos: bool,
) -> ResultadoVentana:
    y_train = y[ventana.inicio_train : ventana.fin_train]
    y_real = y[ventana.inicio_val : ventana.fin_val]
    seed_efectiva = derivar_semilla(seed, identificador, ventana.indice)

    inicio = time.perf_counter()
    try:
        modelo = fabrica(configuracion, seed=seed_efectiva)
        modelo.fit(y_train)
        y_pred = np.asarray(modelo.predict(ventana.horizonte), dtype=float)
        metricas = calcular_metricas(
            y_real, y_pred, y_train=y_train, estacionalidad=estacionalidad
        )
        fallo = None
    except Exception as exc:
        if not tolerar_fallos:
            raise
        y_pred = np.full(ventana.horizonte, np.nan)
        metricas = dict(_METRICAS_FALLO)
        fallo = f"{type(exc).__name__}: {exc}"
    duracion = time.perf_counter() - inicio

    return ResultadoVentana(
        ventana=ventana,
        metricas=metricas,
        y_train=y_train,
        y_real=y_real,
        y_pred=y_pred,
        duracion_s=duracion,
        fallo=fallo,
    )
