"""Precondicion: SKU sin demanda positiva no entra a clasificacion."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from pred_engine.ingesta.salida import HandoffPreconditionError, require_positive_demand


def test_acepta_sku_con_al_menos_un_positivo() -> None:
    panel = pd.DataFrame(
        {
            "sku_id": ["A", "A"],
            "timestamp": pd.date_range(datetime(2024, 10, 1), periods=2, freq="D"),
            "demand_qty": [0.0, 3.0],
            "lead_time_days": [2, 2],
        }
    )
    require_positive_demand(panel)


def test_rechaza_sku_solo_ceros() -> None:
    panel = pd.DataFrame(
        {
            "sku_id": ["Z", "Z"],
            "timestamp": pd.date_range(datetime(2024, 10, 1), periods=2, freq="D"),
            "demand_qty": [0.0, 0.0],
            "lead_time_days": [2, 2],
        }
    )
    with pytest.raises(HandoffPreconditionError, match="sin demanda"):
        require_positive_demand(panel)


def test_panel_vacio_lanza() -> None:
    with pytest.raises(HandoffPreconditionError, match="filas"):
        require_positive_demand(pd.DataFrame())
