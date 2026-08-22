"""Pipeline 1.2 con CSV falso estilo inventory_data (sin red)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pred_engine.ingesta.pipeline import run_ingest


class FakeLlmProvider:
    def complete(self, prompt: str, *, temperature: float, timeout: float) -> str:
        return json.dumps(
            {
                "sku_id": "Item_ID",
                "timestamp": "Date",
                "demand_qty": "Avg_Usage_Per_Day",
                "lead_time_days": "Restock_Lead_Time",
            }
        )


def test_run_ingest_csv_con_huecos(tmp_path: Path) -> None:
    csv = tmp_path / "mini.csv"
    csv.write_text(
        "Date,Item_ID,Item_Name,Current_Stock,Avg_Usage_Per_Day,Restock_Lead_Time\n"
        "2024-10-01,105,Ventilator,1542,108,17\n"
        "2024-10-04,105,Ventilator,900,50,17\n",
        encoding="utf-8",
    )
    raiz = tmp_path / "data"
    resultado = run_ingest(csv, FakeLlmProvider(), data_root=raiz, timeout=5.0)
    assert resultado.parquet_path.is_file()
    panel = pd.read_parquet(resultado.parquet_path)
    assert list(panel.columns) == [
        "sku_id",
        "timestamp",
        "demand_qty",
        "lead_time_days",
    ]
    assert len(panel) == 4
    assert list(panel["demand_qty"]) == [108.0, 0.0, 0.0, 50.0]
    assert "Current_Stock" not in panel.columns
