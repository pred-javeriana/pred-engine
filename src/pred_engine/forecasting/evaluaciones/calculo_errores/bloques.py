"""Bloques de seleccion y de prueba de la reserva (ADR-023).

Con ``R`` dias reservados, horizonte ``H`` y fraccion ``f``, los primeros
``S = floor(f * R)`` dias forman el bloque de seleccion. El origen ``t* + k``
pronostica los dias ``k + 1 .. k + H``:

- seleccion: ``k = 0 .. S - H``, ventanas con todos sus dias en ``1 .. S``;
- embargo: ``k = S - H + 1 .. S - 1``, ventanas que cruzan el limite; nada las usa;
- prueba: ``k = S .. R - H``, ventanas con todos sus dias en ``S + 1 .. R``.

La seleccion de la familia campeona solo lee el bloque de seleccion. Los
veredictos, las pruebas y los efectos solo leen el bloque de prueba.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from pred_engine.comun.reserva import ReserveCut

Bloque = Literal["seleccion", "embargo", "prueba"]


@dataclass(frozen=True, slots=True)
class BloquesReserva:
    """Limites de los bloques, fijados con la politica antes de abrir la reserva."""

    t_star: pd.Timestamp
    dias_reserva: int
    horizonte: int
    fraccion_seleccion: float

    def __post_init__(self) -> None:
        for nombre in ("dias_reserva", "horizonte"):
            valor = getattr(self, nombre)
            if type(valor) is not int or valor < 1:
                raise ValueError(f"{nombre} debe ser un entero >= 1")
        if not 0 < self.fraccion_seleccion < 1:
            raise ValueError("fraccion_seleccion debe estar en (0, 1)")

    @classmethod
    def de(
        cls, corte: ReserveCut, horizonte: int, fraccion_seleccion: float
    ) -> BloquesReserva:
        return cls(corte.t_star, corte.reserved_days, horizonte, fraccion_seleccion)

    @property
    def dias_seleccion(self) -> int:
        # round: 0.4 * 338 es 135.20000000000002 en coma flotante.
        return math.floor(round(self.fraccion_seleccion * self.dias_reserva, 9))

    def bloque(self, origen: pd.Timestamp) -> Bloque:
        k = (origen - self.t_star).days
        if k <= self.dias_seleccion - self.horizonte:
            return "seleccion"
        if k < self.dias_seleccion:
            return "embargo"
        return "prueba"

    def origenes(self, bloque: Bloque) -> pd.DatetimeIndex:
        """Origenes diarios del bloque, en orden; vacio si el bloque no cabe."""
        primero, ultimo = {
            "seleccion": (0, self.dias_seleccion - self.horizonte),
            "embargo": (
                max(0, self.dias_seleccion - self.horizonte + 1),
                self.dias_seleccion - 1,
            ),
            "prueba": (self.dias_seleccion, self.dias_reserva - self.horizonte),
        }[bloque]
        return pd.date_range(
            self.t_star + pd.Timedelta(days=primero),
            periods=max(0, ultimo - primero + 1),
            freq="D",
        )

    def dias(self, bloque: Literal["seleccion", "prueba"]) -> pd.DatetimeIndex:
        """Dias reservados que puntuan las ventanas del bloque."""
        inicio, n = (
            (1, self.dias_seleccion)
            if bloque == "seleccion"
            else (self.dias_seleccion + 1, self.dias_reserva - self.dias_seleccion)
        )
        return pd.date_range(
            self.t_star + pd.Timedelta(days=inicio), periods=n, freq="D"
        )

    def limites(self) -> dict[str, Any]:
        """Parametros y fechas de cada bloque, como los guarda el protocolo de M3.

        ``origenes_*`` son los origenes de cada bloque y ``fechas_*`` los dias
        reservados que puntuan sus ventanas.
        """
        limites: dict[str, Any] = {
            "fraccion_seleccion": self.fraccion_seleccion,
            "dias_reserva": self.dias_reserva,
            "horizonte": self.horizonte,
            "dias_seleccion": self.dias_seleccion,
        }
        for bloque in ("seleccion", "embargo", "prueba"):
            limites[f"origenes_{bloque}"] = _rango(self.origenes(bloque))
        for bloque in ("seleccion", "prueba"):
            limites[f"fechas_{bloque}"] = _rango(self.dias(bloque))
        return limites


def _rango(fechas: pd.DatetimeIndex) -> dict[str, Any]:
    iso = list(fechas.strftime("%Y-%m-%d"))
    return {
        "n": len(iso),
        "primero": iso[0] if iso else None,
        "ultimo": iso[-1] if iso else None,
    }
