"""Corte cronologico de la reserva para M3 (ADR-03-003, 3.1).

M1 clasifica cada SKU solo con la historia anterior al corte (ADR-019) y M2
optimiza y ajusta con esa misma historia; los dias reservados quedan para la
evaluacion de M3. Ambos modulos derivan el corte del calendario del panel con
esta misma funcion, asi que coinciden sin intercambiar estado.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import pandas as pd

# ADR-03-003: el 20 % final y contiguo del calendario del panel queda reservado
# para M3. La fraccion es fija; no se ajusta por SKU ni por corrida.
RESERVE_FRACTION = 0.2


@dataclass(frozen=True, slots=True)
class ReserveCut:
    """Corte cronologico unificado (ADR-03-003, 3.1): ni M1 ni M2 leen despues de t*.

    La reserva son los ultimos ``reserved_days`` dias del calendario del panel,
    ``ceil(fraction * dias)``, desde ``first_reserved`` hasta ``last_observed``.
    """

    t_star: pd.Timestamp
    first_reserved: pd.Timestamp
    last_observed: pd.Timestamp
    reserved_days: int
    fraction: float = RESERVE_FRACTION

    @classmethod
    def of(cls, panel: pd.DataFrame, fraction: float = RESERVE_FRACTION) -> ReserveCut:
        if not 0 < fraction < 1:
            raise ValueError("la fraccion reservada debe estar en (0, 1)")
        stamps = pd.to_datetime(panel["timestamp"]).dt.normalize()
        first, last = stamps.min(), stamps.max()
        total_days = (last - first).days + 1
        # round: 0.2 * 35 es 7.000000000000001 en coma flotante, no 8 dias.
        reserved = math.ceil(round(fraction * total_days, 9))
        if reserved >= total_days:
            raise ValueError(
                f"el panel cubre {total_days} dias: no deja historia admisible "
                f"despues de reservar {reserved}"
            )
        t_star = cast(pd.Timestamp, last - pd.Timedelta(days=reserved))
        return cls(
            t_star=t_star,
            first_reserved=cast(pd.Timestamp, t_star + pd.Timedelta(days=1)),
            last_observed=last,
            reserved_days=reserved,
            fraction=fraction,
        )

    @property
    def reserved_dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.first_reserved, periods=self.reserved_days, freq="D")
