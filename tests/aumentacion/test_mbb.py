"""Pruebas de las primitivas de Moving Block Bootstrap del companiero (mbb.py).

La Fase 0 se apoya en estas funciones; se cubren aqui sus contratos y guardas.
"""

from __future__ import annotations

import numpy as np
import pytest

from pred_engine.aumentacion.mbb import (
    aumentar,
    compose_series,
    decompose_series,
    moving_block_bootstrap,
)


def _serie_estacional(n: int = 120, period: int = 7) -> np.ndarray:
    base = np.arange(n)
    rng = np.random.default_rng(0)
    return (
        20.0
        + 0.03 * base
        + 4.0 * np.sin(2 * np.pi * base / period)
        + rng.normal(0.0, 1.0, n)
    )


def test_decompose_reconstruye_la_serie() -> None:
    serie = _serie_estacional()
    trend, seasonal, residual = decompose_series(serie, period=7)
    reconstruida = compose_series(trend, seasonal, residual)
    assert np.allclose(reconstruida, serie, atol=1e-6)


def test_decompose_rechaza_series_invalidas() -> None:
    with pytest.raises(ValueError):
        decompose_series(np.zeros((2, 2)), period=7)
    with pytest.raises(ValueError):
        decompose_series(np.array([]), period=7)
    with pytest.raises(ValueError):
        decompose_series(np.ones(100), period=1)
    with pytest.raises(ValueError):
        decompose_series(np.ones(5), period=7)


def test_compose_exige_misma_longitud() -> None:
    with pytest.raises(ValueError):
        compose_series(np.ones(3), np.ones(3), np.ones(4))
    with pytest.raises(ValueError):
        compose_series(np.ones((2, 2)), np.ones(4), np.ones(4))


def test_moving_block_bootstrap_conserva_longitud_y_es_reproducible() -> None:
    serie = _serie_estacional()
    a = moving_block_bootstrap(
        serie, block_size=5, tamaño_esperado=len(serie), rng=np.random.default_rng(1)
    )
    b = moving_block_bootstrap(
        serie, block_size=5, tamaño_esperado=len(serie), rng=np.random.default_rng(1)
    )
    assert a.shape == serie.shape
    assert np.array_equal(a, b)


def test_moving_block_bootstrap_valida_argumentos() -> None:
    serie = np.ones(10)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        moving_block_bootstrap(serie, 0, 10, rng)
    with pytest.raises(ValueError):
        moving_block_bootstrap(serie, 20, 10, rng)
    with pytest.raises(ValueError):
        moving_block_bootstrap(serie, 3, -1, rng)
    with pytest.raises(ValueError):
        moving_block_bootstrap(np.array([]), 3, 10, rng)
    assert moving_block_bootstrap(serie, 3, 0, rng).size == 0


def test_aumentar_devuelve_original_mas_sinteticas() -> None:
    serie = _serie_estacional()
    salida = aumentar(serie, period=7, n_series=3)
    assert len(salida) == 4
    assert np.array_equal(salida[0], serie)
    assert all(s.shape == serie.shape for s in salida[1:])


def test_aumentar_rechaza_n_series_no_positivo() -> None:
    with pytest.raises(ValueError):
        aumentar(_serie_estacional(), period=7, n_series=0)
