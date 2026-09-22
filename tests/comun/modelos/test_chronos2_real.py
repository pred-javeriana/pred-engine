"""Chronos-2 real en CPU: determinismo y pesos congelados.

Requiere el extra `foundation` y los pesos (se descargan una vez a la cache de
Hugging Face). Reproducible con:

    uv sync --extra dev --extra foundation
    uv run pytest tests/comun/modelos/test_chronos2_real.py -m slow -q
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("chronos")
torch = pytest.importorskip("torch")

from pred_engine.comun.modelos.modelos_fundacionales import (  # noqa: E402
    CHRONOS2_ZERO_SHOT,
    Chronos2Forecaster,
    cargar_pipeline,
)

pytestmark = pytest.mark.slow


def _serie_intermitente(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n) < 0.3, rng.poisson(5, n), 0).astype(float)


def _pesos():
    pipeline = cargar_pipeline(CHRONOS2_ZERO_SHOT)
    modelo = pipeline._pipeline.model  # type: ignore[attr-defined]
    return {k: v.detach().clone() for k, v in modelo.state_dict().items()}


def test_dos_corridas_dan_el_mismo_pronostico_bit_a_bit():
    y = _serie_intermitente()
    a = Chronos2Forecaster().fit(y).predict_cuantiles(14)
    b = Chronos2Forecaster().fit(y).predict_cuantiles(14)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (21, 14)
    assert np.all(a >= 0)


def test_predecir_no_modifica_los_pesos():
    antes = _pesos()
    Chronos2Forecaster().fit(_serie_intermitente()).predict(28)
    despues = _pesos()
    assert antes.keys() == despues.keys()
    assert all(torch.equal(antes[k], despues[k]) for k in antes)
