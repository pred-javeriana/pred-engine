"""Moving block bootstrap para aumentacion de series temporales."""

from __future__ import annotations

import numpy as np
from statsmodels.tsa.seasonal import STL


def aumentar(
    series_array: np.ndarray,
    period: int,
    n_series: int = 1,
    block_size: int = 3,
) -> list[np.ndarray]:
    if n_series <= 0:
        raise ValueError("n_series debe ser mayor que 0.")

    trend, seasonal, residual = decompose_series(
        series_array=series_array,
        period=period,
    )
    rng = np.random.default_rng(42)
    synthetic_series = [series_array.copy()]
    for _ in range(n_series):
        bootstrap_residual = moving_block_bootstrap(
            serie_sku=residual,
            block_size=block_size,
            tamaño_esperado=len(residual),
            rng=rng,
        )
        synthetic_series.append(
            compose_series(
                trend=trend,
                seasonal=seasonal,
                residual=bootstrap_residual,
            )
        )

    return synthetic_series


def decompose_series(
    series_array: np.ndarray,
    period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if series_array.ndim != 1:
        raise ValueError("series_array debe ser un array 1D.")

    if series_array.size == 0:
        raise ValueError("series_array no puede estar vacío.")

    if period <= 1:
        raise ValueError("period debe ser mayor que 1.")

    if series_array.size < 2 * period:
        raise ValueError("La serie es demasiado corta para el periodo indicado.")

    result = STL(
        series_array,
        period=period,
    ).fit()

    trend = np.asarray(result.trend)
    seasonal = np.asarray(result.seasonal)
    residual = np.asarray(result.resid)

    return trend, seasonal, residual


def compose_series(
    trend: np.ndarray,
    seasonal: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    if trend.ndim != 1:
        raise ValueError("trend debe ser un array 1D.")

    if seasonal.ndim != 1:
        raise ValueError("seasonal debe ser un array 1D.")

    if residual.ndim != 1:
        raise ValueError("residual debe ser un array 1D.")

    if not (len(trend) == len(seasonal) == len(residual)):
        raise ValueError("Todos los componentes deben tener la misma longitud.")

    return trend + seasonal + residual


def moving_block_bootstrap(
    serie_sku: np.ndarray,
    block_size: int,
    tamaño_esperado: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera una serie aumentada mediante moving block bootstrap."""
    if serie_sku.ndim != 1:
        raise ValueError("serie_sku debe ser un array 1D.")

    if serie_sku.size == 0:
        raise ValueError("serie_sku no puede estar vacío.")

    if block_size <= 0:
        raise ValueError("block_size debe ser mayor que 0.")

    if block_size > serie_sku.size:
        raise ValueError("block_size no puede ser mayor que la longitud de la serie.")

    if tamaño_esperado < 0:
        raise ValueError("tamaño_esperado no puede ser negativo.")

    if tamaño_esperado == 0:
        return np.empty(0, dtype=serie_sku.dtype)

    n = serie_sku.size
    num_blocks = int(np.ceil(tamaño_esperado / block_size))
    max_start = n - block_size

    i_bloques = rng.integers(0, max_start + 1, size=num_blocks)
    bloques = [serie_sku[i : i + block_size] for i in i_bloques]
    serie_aumentada = np.concatenate(bloques)

    return serie_aumentada[:tamaño_esperado]
