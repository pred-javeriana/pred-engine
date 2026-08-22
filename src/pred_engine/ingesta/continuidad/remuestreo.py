"""Remuestreo diario por SKU con zero-filling de demanda."""

from __future__ import annotations

import pandas as pd

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos import CANONICAL_FIELDS
from pred_engine.ingesta.continuidad.errores import TemporalContinuityError

_logger = get_logger(__name__)


def resample_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Left join sobre un grid diario por sku_id. Imputa demanda 0 en huecos."""
    if frame.empty:
        raise TemporalContinuityError("no hay filas para remuestrear")
    faltan = [c for c in CANONICAL_FIELDS if c not in frame.columns]
    if faltan:
        raise TemporalContinuityError(f"faltan columnas {faltan}")

    piezas: list[pd.DataFrame] = []
    for sku, grupo in frame.groupby("sku_id", sort=True):
        grupo = grupo.sort_values("timestamp").copy()
        inicio = pd.Timestamp(grupo["timestamp"].min()).normalize()
        fin = pd.Timestamp(grupo["timestamp"].max()).normalize()
        grid = pd.DataFrame(
            {
                "sku_id": sku,
                "timestamp": pd.date_range(inicio, fin, freq="D"),
            }
        )
        fusion = grid.merge(
            grupo,
            on=["sku_id", "timestamp"],
            how="left",
        )
        n_huecos = int(fusion["demand_qty"].isna().sum())
        fusion["demand_qty"] = fusion["demand_qty"].fillna(0.0)
        fusion["lead_time_days"] = fusion["lead_time_days"].ffill().bfill()
        if fusion["lead_time_days"].isna().any():
            raise TemporalContinuityError(
                f"SKU {sku!r} no tiene lead_time_days observable para imputar"
            )
        fusion["lead_time_days"] = fusion["lead_time_days"].astype("int64")
        fusion["demand_qty"] = fusion["demand_qty"].astype("float64")
        _logger.info(
            "Remuestreo SKU=%s dias=%s huecos_demanda=%s",
            sku,
            len(fusion),
            n_huecos,
        )
        piezas.append(fusion.loc[:, list(CANONICAL_FIELDS)])

    panel = pd.concat(piezas, ignore_index=True)
    deltas = panel.groupby("sku_id")["timestamp"].diff().dropna()
    if not deltas.empty and not bool((deltas == pd.Timedelta(days=1)).all()):
        # El primer diff por grupo es NaT (ya dropna); el resto debe ser 1 dia.
        raise TemporalContinuityError("la cuadrícula no es estrictamente diaria")
    return panel
