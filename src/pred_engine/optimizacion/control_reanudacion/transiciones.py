"""Maquina de estados de la corrida PRED.

La transicion identidad se permite para que un checkpoint repetido sobre
`EN_PROGRESO` (o una reapertura de `COMPLETADA` que no reejecuta) sea
idempotente.
"""

from __future__ import annotations

from collections.abc import Mapping

from pred_engine.optimizacion.control_reanudacion.contratos import EstadoCorrida
from pred_engine.optimizacion.control_reanudacion.errores import TransicionEstadoError

TRANSICIONES_PERMITIDAS: Mapping[EstadoCorrida, frozenset[EstadoCorrida]] = {
    EstadoCorrida.NUEVA: frozenset({EstadoCorrida.EN_PROGRESO}),
    EstadoCorrida.EN_PROGRESO: frozenset(
        {
            EstadoCorrida.EN_PROGRESO,
            EstadoCorrida.COMPLETADA,
            EstadoCorrida.FALLIDA,
            EstadoCorrida.INTERRUMPIDA,
        }
    ),
    EstadoCorrida.INTERRUMPIDA: frozenset(
        {EstadoCorrida.EN_PROGRESO, EstadoCorrida.FALLIDA}
    ),
    EstadoCorrida.COMPLETADA: frozenset({EstadoCorrida.COMPLETADA}),
    EstadoCorrida.FALLIDA: frozenset(),
}


def transicionar(actual: EstadoCorrida, nuevo: EstadoCorrida) -> EstadoCorrida:
    permitidos = TRANSICIONES_PERMITIDAS[actual]
    if nuevo not in permitidos:
        raise TransicionEstadoError(
            f"transicion invalida: {actual.value} -> {nuevo.value}"
        )
    return nuevo
