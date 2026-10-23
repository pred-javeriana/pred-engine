"""Productos sin demanda positiva en la historia admisible de M1."""

from dataclasses import dataclass

import pandas as pd

from pred_engine.comun.reserva import ReserveCut


@dataclass(frozen=True, slots=True)
class SkuExclusion:
    """SKU descartado antes de L2 con su causa registrada."""

    sku_id: str
    cause: str


def sin_demanda_admisible(
    panel: pd.DataFrame, corte: ReserveCut
) -> tuple[SkuExclusion, ...]:
    """SKU sin demanda positiva hasta el ``corte`` de la corrida completa."""
    dias = pd.to_datetime(panel["timestamp"]).dt.normalize()
    positivos = panel.loc[
        (dias <= corte.t_star) & (panel["demand_qty"] > 0), "sku_id"
    ].astype("string")
    todos = panel["sku_id"].astype("string")
    return tuple(
        SkuExclusion(str(sku), "lanzado despues del corte")
        for sku in sorted(set(todos.dropna()) - set(positivos.dropna()))
    )
