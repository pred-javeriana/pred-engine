"""Huecos diarios → 0; ninguna fecha omitida en el rango del SKU."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from pred_engine.ingesta.continuidad import resample_daily


def _marco() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": pd.Series(["A", "A", "B"], dtype="string"),
            "timestamp": pd.to_datetime(
                [
                    datetime(2024, 10, 1),
                    datetime(2024, 10, 4),
                    datetime(2024, 10, 1),
                ]
            ),
            "demand_qty": [10.0, 7.0, 1.0],
            "lead_time_days": [5, 8, 3],
        }
    )


def test_grid_diario_sin_fechas_omitidas() -> None:
    panel = resample_daily(_marco())
    sku_a = panel[panel["sku_id"] == "A"].sort_values("timestamp")
    assert len(sku_a) == 4
    assert list(sku_a["timestamp"].dt.day) == [1, 2, 3, 4]
    diffs = sku_a["timestamp"].diff().dropna()
    assert (diffs == pd.Timedelta(days=1)).all()


def test_huecos_quedan_en_cero() -> None:
    panel = resample_daily(_marco())
    sku_a = panel[panel["sku_id"] == "A"].sort_values("timestamp")
    assert list(sku_a["demand_qty"]) == [10.0, 0.0, 0.0, 7.0]


def test_sku_sin_huecos_no_se_infla() -> None:
    panel = resample_daily(_marco())
    sku_b = panel[panel["sku_id"] == "B"]
    assert len(sku_b) == 1
    assert sku_b.iloc[0]["demand_qty"] == 1.0


def test_lead_time_se_arrastra_en_huecos() -> None:
    panel = resample_daily(_marco())
    sku_a = panel[panel["sku_id"] == "A"].sort_values("timestamp")
    assert sku_a.iloc[1]["lead_time_days"] == 5
