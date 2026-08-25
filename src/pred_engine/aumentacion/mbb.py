"""Moving block bootstrap para aumentacion de series temporales."""

from __future__ import annotations

import numpy as np


def moving_block_bootstrap(
    serie_sku: np.ndarray,
    block_size: int,
    tamaño_esperado: int,
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

    rng = np.random.default_rng(42)
    n = serie_sku.size
    num_blocks = int(np.ceil(tamaño_esperado / block_size))
    max_start = n - block_size

    i_bloques = rng.integers(0, max_start + 1, size=num_blocks)
    bloques = [serie_sku[i : i + block_size] for i in i_bloques]
    serie_aumentada = np.concatenate(bloques)

    return serie_aumentada[:tamaño_esperado]
