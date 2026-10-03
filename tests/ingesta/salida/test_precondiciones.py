"""Precondicion: SKU sin demanda positiva no entra a clasificacion."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from pred_engine.ingesta.salida import HandoffPreconditionError, require_positive_demand


def _panel(sku: str, demanda: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": [sku] * len(demanda),
            "timestamp": pd.date_range(
                datetime(2024, 10, 1), periods=len(demanda), freq="D"
            ),
            "demand_qty": demanda,
            "lead_time_days": [2] * len(demanda),
        }
    )


def test_acepta_sku_con_al_menos_un_positivo_antes_de_t_estrella() -> None:
    require_positive_demand(_panel("A", [0.0, 3.0, 0.0]))


def test_rechaza_sku_con_demanda_solo_en_la_reserva() -> None:
    # 5 dias: el ultimo es la reserva; antes de t* no hay demanda.
    with pytest.raises(HandoffPreconditionError, match=r"hasta t\*=2024-10-04: R"):
        require_positive_demand(_panel("R", [0.0, 0.0, 0.0, 0.0, 7.0]))


def test_rechaza_panel_sin_historia_admisible() -> None:
    with pytest.raises(HandoffPreconditionError, match="historia admisible"):
        require_positive_demand(_panel("U", [4.0]))


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
