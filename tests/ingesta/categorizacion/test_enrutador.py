"""Umbrales 1.32 / 0.49 y empates al cuadrante alto."""

from __future__ import annotations

import math

import pytest

from pred_engine.ingesta.categorizacion import (
    TopologyRoutingError,
    route_syntetos_boylan,
)


def test_suave() -> None:
    assert route_syntetos_boylan(1.0, 0.2) == "smooth"


def test_intermitente() -> None:
    assert route_syntetos_boylan(2.0, 0.2) == "intermittent"


def test_erratica() -> None:
    assert route_syntetos_boylan(1.0, 0.8) == "erratic"


def test_lumpy() -> None:
    assert route_syntetos_boylan(2.0, 0.8) == "lumpy"


def test_empate_adi_132_cv_bajo_es_intermitente() -> None:
    assert route_syntetos_boylan(1.32, 0.48) == "intermittent"


def test_empate_cv2_049_adi_bajo_es_erratica() -> None:
    assert route_syntetos_boylan(1.31, 0.49) == "erratic"


def test_empate_ambos_umbrales_es_lumpy() -> None:
    assert route_syntetos_boylan(1.32, 0.49) == "lumpy"


def test_justo_bajo_ambos_es_suave() -> None:
    assert route_syntetos_boylan(1.319999, 0.489999) == "smooth"


def test_no_finitos_lanzan() -> None:
    with pytest.raises(TopologyRoutingError):
        route_syntetos_boylan(math.inf, 0.1)
    with pytest.raises(TopologyRoutingError):
        route_syntetos_boylan(1.0, math.nan)
