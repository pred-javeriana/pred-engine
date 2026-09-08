"""Metricas de error y su agregacion segura entre ventanas.

Convenciones deliberadas (documentadas aqui porque no son obvias):

* ``smape``: el denominador ``(|y|+|y_hat|)/2`` es 0 cuando real y predicho
  son ambos 0 -- constante en SKUs intermitentes/lumpy. Convencion: ese
  termino aporta 0 al promedio (no NaN, no se descarta la observacion).
* ``mase``: el denominador SIEMPRE se calcula sobre el ``y_train`` de esa
  misma ventana, nunca sobre la serie completa -- calcularlo sobre la serie
  completa metería informacion del futuro dentro de la metrica de esa
  ventana. Si el denominador es 0 (naive estacional de train es constante),
  MASE es ``inf``, nunca ``0.0`` (que se leeria como "modelo perfecto").
* ``agregar``: implementa la regla de seguridad #2 de la poda (comparar
  configuraciones por media/mediana recortada, nunca por una ventana
  aislada). La usan tanto el modo integro como el greedy (via
  `valor_agregado_de`), para que sus resultados sean comparables.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np

from pred_engine.comun.dataclasses.validacion_temporal import ResultadoVentana

_ESTRATEGIAS_AGREGACION = ("media", "mediana", "media_recortada")


def mae(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_real - y_pred)))


def rmse(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_real - y_pred) ** 2)))


def smape(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominador = (np.abs(y_real) + np.abs(y_pred)) / 2.0
    numerador = np.abs(y_real - y_pred)
    terminos = np.divide(
        numerador,
        denominador,
        out=np.zeros_like(numerador),
        where=denominador != 0,
    )
    return float(np.mean(terminos) * 100.0)


def mase(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_train: np.ndarray,
    estacionalidad: int = 1,
) -> float:
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if estacionalidad < 1:
        raise ValueError("estacionalidad debe ser >= 1")
    if len(y_train) <= estacionalidad:
        return float("inf")
    diffs_naive = np.abs(y_train[estacionalidad:] - y_train[:-estacionalidad])
    denominador = float(np.mean(diffs_naive))
    if not math.isfinite(denominador) or denominador == 0.0:
        return float("inf")
    return float(np.mean(np.abs(y_real - y_pred)) / denominador)


def calcular_metricas(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_train: np.ndarray,
    estacionalidad: int = 1,
) -> dict[str, float]:
    return {
        "mae": mae(y_real, y_pred),
        "rmse": rmse(y_real, y_pred),
        "smape": smape(y_real, y_pred),
        "mase": mase(y_real, y_pred, y_train=y_train, estacionalidad=estacionalidad),
    }


def agregar(
    valores: Sequence[float],
    *,
    estrategia: Literal["media", "mediana", "media_recortada"] = "media_recortada",
    proporcion_recorte: float = 0.1,
) -> float:
    if not valores:
        raise ValueError("valores no puede estar vacio")
    if estrategia not in _ESTRATEGIAS_AGREGACION:
        raise ValueError(f"estrategia debe ser una de {_ESTRATEGIAS_AGREGACION}")

    arreglo = np.asarray(list(valores), dtype=float)
    finitos = arreglo[np.isfinite(arreglo)]
    if finitos.size == 0:
        return float("inf")

    if estrategia == "media":
        return float(np.mean(finitos))
    if estrategia == "mediana":
        return float(np.median(finitos))

    if not 0.0 <= proporcion_recorte < 0.5:
        raise ValueError("proporcion_recorte debe estar en [0, 0.5)")
    ordenado = np.sort(finitos)
    n = len(ordenado)
    recorte = int(math.floor(n * proporcion_recorte))
    if recorte <= 0 or recorte * 2 >= n:
        return float(np.mean(ordenado))
    return float(np.mean(ordenado[recorte : n - recorte]))


def valor_agregado_de(
    resultados: Sequence[ResultadoVentana],
    metrica_objetivo: str,
    *,
    estrategia: Literal["media", "mediana", "media_recortada"] = "media_recortada",
    proporcion_recorte: float = 0.1,
) -> float:
    valores = [
        r.metricas[metrica_objetivo]
        for r in resultados
        if r.fallo is None
        and math.isfinite(r.metricas.get(metrica_objetivo, float("nan")))
    ]
    if not valores:
        return float("inf")
    return agregar(
        valores, estrategia=estrategia, proporcion_recorte=proporcion_recorte
    )
