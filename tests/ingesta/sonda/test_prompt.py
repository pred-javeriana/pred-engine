"""El prompt debe pedir diagnostico JSON y prohibir mutacion."""

from __future__ import annotations

import pandas as pd

from pred_engine.ingesta.sonda.prompt import build_alignment_prompt


def test_inyecta_cinco_filas_y_cabeceras_del_baseline() -> None:
    marco = pd.DataFrame(
        {
            "Date": ["2024-10-01", "2024-10-02"],
            "Item_ID": ["105", "100"],
            "Current_Stock": ["1542", "2487"],
            "Avg_Usage_Per_Day": ["108", "55"],
            "Restock_Lead_Time": ["17", "12"],
        }
    )
    prompt = build_alignment_prompt(marco)
    assert "Date,Item_ID" in prompt.replace(" ", "")
    assert "Date" in prompt
    assert "Current_Stock" in prompt
    assert "demand_qty" in prompt
    assert "NEVER treat on-hand stock" in prompt
    assert "NUNCA" in prompt


def test_pide_status_y_diagnostico_sin_mapeo() -> None:
    marco = pd.DataFrame({"sku_id": ["1"], "timestamp": ["2024-01-01"]})
    prompt = build_alignment_prompt(marco)
    assert '"status"' in prompt
    assert '"diagnostic"' in prompt
    assert "Do NOT rename columns" in prompt
    assert "NO renombres columnas" in prompt
    assert "df.rename" not in prompt
