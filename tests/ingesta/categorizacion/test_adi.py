"""ADI ignora no-positivos en el numerador activo y evita division por cero."""

from __future__ import annotations

import numpy as np
import pytest

from pred_engine.ingesta.categorizacion import TopologyMathError, compute_adi


def test_adi_serie_densa_es_uno() -> None:
    assert compute_adi([5.0, 5.0, 5.0, 5.0]) == 1.0


def test_adi_con_ceros() -> None:
    # 7 periodos finitos, 2 positivos → 3.5
    assert compute_adi([5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0]) == pytest.approx(3.5)


def test_adi_ignora_nan_en_periodos() -> None:
    # NaN no es periodo evaluado: 2 finitos / 2 positivos = 1
    assert compute_adi([1.0, np.nan, 2.0]) == pytest.approx(1.0)


def test_adi_negativos_no_son_demanda_activa() -> None:
    # 3 finitos, 2 positivos → 1.5
    assert compute_adi([1.0, -3.0, 2.0]) == pytest.approx(1.5)


def test_adi_cero_positivos_lanza_error_tipado() -> None:
    with pytest.raises(TopologyMathError, match="division por cero"):
        compute_adi([0.0, 0.0, -1.0])


def test_adi_vacio_o_solo_nan_lanza() -> None:
    with pytest.raises(TopologyMathError):
        compute_adi([])
    with pytest.raises(TopologyMathError):
        compute_adi([np.nan, np.nan])


def test_adi_rechaza_2d() -> None:
    with pytest.raises(TopologyMathError, match="1-D"):
        compute_adi([[1.0, 2.0]])
