"""Extras ERP se descartan; el origen no se muta."""

from __future__ import annotations

import pandas as pd
import pytest

from pred_engine.comun.modelos import CANONICAL_FIELDS
from pred_engine.ingesta.categorizacion import (
    TopologyContractError,
    select_canonical_columns,
)


def test_proyeccion_descarta_extras_sin_mutar() -> None:
    origen = pd.DataFrame(
        {
            "sku_id": ["105"],
            "timestamp": ["2024-10-01"],
            "Item_Name": ["Ventilator"],
            "demand_qty": ["108"],
            "lead_time_days": ["17"],
            "Vendor_ID": ["V001"],
        }
    )
    columnas = list(origen.columns)
    proyectado = select_canonical_columns(origen)
    assert list(proyectado.columns) == list(CANONICAL_FIELDS)
    assert list(origen.columns) == columnas
    assert "Item_Name" not in proyectado.columns


def test_proyeccion_falla_si_falta_canonica() -> None:
    marco = pd.DataFrame({"sku_id": ["1"], "timestamp": ["2024-10-01"]})
    with pytest.raises(TopologyContractError, match="faltan"):
        select_canonical_columns(marco)
