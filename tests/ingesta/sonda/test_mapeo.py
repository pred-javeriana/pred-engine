"""Rename + drop sobre cabeceras estilo inventory_data y CSVs hostiles."""

from __future__ import annotations

import pandas as pd
import pytest

from pred_engine.comun.modelos import HeaderMapping
from pred_engine.ingesta.sonda import SemanticAlignmentError, apply_header_mapping


def _marco_baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2024-10-01"],
            "Item_ID": ["105"],
            "Item_Name": ["Ventilator"],
            "Current_Stock": ["1542"],
            "Avg_Usage_Per_Day": ["108"],
            "Restock_Lead_Time": ["17"],
            "Vendor_ID": ["V001"],
        }
    )


def test_mapeo_baseline_descarta_ruido() -> None:
    mapeo = HeaderMapping(
        sku_id="Item_ID",
        timestamp="Date",
        demand_qty="Avg_Usage_Per_Day",
        lead_time_days="Restock_Lead_Time",
    )
    artefacto = apply_header_mapping(_marco_baseline(), mapeo)
    assert list(artefacto.frame.columns) == [
        "sku_id",
        "timestamp",
        "demand_qty",
        "lead_time_days",
    ]
    assert "Current_Stock" in artefacto.dropped_columns
    assert "Vendor_ID" in artefacto.dropped_columns
    assert artefacto.frame.iloc[0]["demand_qty"] == "108"


def test_falla_si_falta_demanda() -> None:
    mapeo = HeaderMapping(
        sku_id="Item_ID",
        timestamp="Date",
        demand_qty=None,
        lead_time_days="Restock_Lead_Time",
    )
    with pytest.raises(SemanticAlignmentError, match="no se identificaron"):
        apply_header_mapping(_marco_baseline(), mapeo)


def test_falla_si_el_llm_inventa_columna() -> None:
    mapeo = HeaderMapping(
        sku_id="Not_A_Column",
        timestamp="Date",
        demand_qty="Avg_Usage_Per_Day",
        lead_time_days="Restock_Lead_Time",
    )
    with pytest.raises(SemanticAlignmentError, match="no existe"):
        apply_header_mapping(_marco_baseline(), mapeo)
