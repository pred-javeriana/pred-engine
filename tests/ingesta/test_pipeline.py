"""Pipeline 1.2 con CSV canonico (sonda aceptada, sin rename automatico)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pred_engine.ingesta.pipeline import run_ingest
from pred_engine.ingesta.sonda import SemanticAlignmentError


class FakeLlmProvider:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def complete(self, prompt: str, *, temperature: float, timeout: float) -> str:
        return json.dumps(self.payload)


_ACCEPTED = {
    "status": "accepted",
    "diagnostic": [
        {
            "field": "schema",
            "severity": "info",
            "message": "Cabeceras canonicas presentes",
            "action": None,
        }
    ],
}

_REJECTED = {
    "status": "rejected",
    "diagnostic": [
        {
            "field": "timestamp",
            "severity": "error",
            "message": "Renombrar Date",
            "action": "Renombrar 'Date' a 'timestamp'",
        }
    ],
}


def test_run_ingest_csv_canonico_con_huecos(tmp_path: Path) -> None:
    csv = tmp_path / "mini.csv"
    csv.write_text(
        "sku_id,timestamp,demand_qty,lead_time_days\n"
        "105,2024-10-01,108,17\n"
        "105,2024-10-04,50,17\n",
        encoding="utf-8",
    )
    raiz = tmp_path / "data"
    resultado = run_ingest(csv, FakeLlmProvider(_ACCEPTED), data_root=raiz, timeout=5.0)
    assert resultado.parquet_path.is_file()
    assert resultado.diagnostic.diagnostic.is_accepted()
    assert list(resultado.diagnostic.frame.columns) == [
        "sku_id",
        "timestamp",
        "demand_qty",
        "lead_time_days",
    ]
    panel = pd.read_parquet(resultado.parquet_path)
    assert list(panel.columns) == [
        "sku_id",
        "timestamp",
        "demand_qty",
        "lead_time_days",
    ]
    assert len(panel) == 4
    assert list(panel["demand_qty"]) == [108.0, 0.0, 0.0, 50.0]


def test_run_ingest_rechaza_csv_no_canonico_sin_mutar(tmp_path: Path) -> None:
    csv = tmp_path / "hostil.csv"
    csv.write_text(
        "Date,Item_ID,Avg_Usage_Per_Day,Restock_Lead_Time\n2024-10-01,105,108,17\n",
        encoding="utf-8",
    )
    raiz = tmp_path / "data"
    with pytest.raises(SemanticAlignmentError) as exc:
        run_ingest(csv, FakeLlmProvider(_REJECTED), data_root=raiz, timeout=5.0)
    assert exc.value.diagnostic is not None
    assert exc.value.diagnostic.is_rejected()
