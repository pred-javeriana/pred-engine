"""CV² excluye ceros y devuelve float; un solo positivo es variacion nula."""

from __future__ import annotations

import numpy as np
import pytest

from pred_engine.ingesta.categorizacion import TopologyMathError, compute_cv2


def test_cv2_constante_es_cero() -> None:
    assert compute_cv2([5.0, 5.0, 5.0, 5.0]) == pytest.approx(0.0)


def test_cv2_excluye_ceros() -> None:
    # Solo [5, 5]; ceros no entran a μ ni σ
    assert compute_cv2([5.0, 0.0, 0.0, 5.0]) == pytest.approx(0.0)


def test_cv2_un_positivo_es_cero() -> None:
    assert compute_cv2([0.0, 5.0, 0.0]) == 0.0


def test_cv2_erratico_conocido() -> None:
    valores = [1.0, 10.0, 2.0, 20.0]
    media = float(np.mean(valores))
    sigma = float(np.std(valores, ddof=1))
    esperado = (sigma / media) ** 2
    assert compute_cv2(valores) == pytest.approx(esperado)
    assert isinstance(compute_cv2(valores), float)


def test_cv2_sin_positivos_lanza() -> None:
    with pytest.raises(TopologyMathError, match="estrictamente positiva"):
        compute_cv2([0.0, 0.0, -2.0, np.nan])


def test_cv2_rechaza_2d() -> None:
    with pytest.raises(TopologyMathError, match="1-D"):
        compute_cv2([[1.0], [2.0]])
