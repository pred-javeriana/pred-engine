"""Transiciones 2.9-A1: maquina de estados fail-closed e identitaria."""

from __future__ import annotations

import pytest

from pred_engine.optimizacion.control_reanudacion import EstadoCorrida
from pred_engine.optimizacion.control_reanudacion.errores import TransicionEstadoError
from pred_engine.optimizacion.control_reanudacion.transiciones import transicionar


@pytest.mark.parametrize(
    ("origen", "destino"),
    [
        (EstadoCorrida.NUEVA, EstadoCorrida.EN_PROGRESO),
        (EstadoCorrida.EN_PROGRESO, EstadoCorrida.COMPLETADA),
        (EstadoCorrida.EN_PROGRESO, EstadoCorrida.FALLIDA),
        (EstadoCorrida.EN_PROGRESO, EstadoCorrida.INTERRUMPIDA),
        (EstadoCorrida.INTERRUMPIDA, EstadoCorrida.EN_PROGRESO),
        (EstadoCorrida.INTERRUMPIDA, EstadoCorrida.FALLIDA),
        (EstadoCorrida.EN_PROGRESO, EstadoCorrida.EN_PROGRESO),
        (EstadoCorrida.COMPLETADA, EstadoCorrida.COMPLETADA),
    ],
)
def test_transiciones_permitidas(origen: EstadoCorrida, destino: EstadoCorrida) -> None:
    assert transicionar(origen, destino) is destino


@pytest.mark.parametrize(
    ("origen", "destino"),
    [
        (EstadoCorrida.NUEVA, EstadoCorrida.COMPLETADA),
        (EstadoCorrida.NUEVA, EstadoCorrida.INTERRUMPIDA),
        (EstadoCorrida.COMPLETADA, EstadoCorrida.EN_PROGRESO),
        (EstadoCorrida.COMPLETADA, EstadoCorrida.NUEVA),
        (EstadoCorrida.FALLIDA, EstadoCorrida.EN_PROGRESO),
        (EstadoCorrida.INTERRUMPIDA, EstadoCorrida.COMPLETADA),
        (EstadoCorrida.EN_PROGRESO, EstadoCorrida.NUEVA),
    ],
)
def test_transiciones_ilegales(origen: EstadoCorrida, destino: EstadoCorrida) -> None:
    with pytest.raises(TransicionEstadoError, match="transicion invalida"):
        transicionar(origen, destino)
