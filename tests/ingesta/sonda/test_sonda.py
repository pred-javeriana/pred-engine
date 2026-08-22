"""Sonda con proveedor falso: exito, timeout y JSON basura."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pred_engine.comun.llm import LlmTimeoutError
from pred_engine.ingesta.sonda import (
    SemanticAlignmentError,
    parse_header_mapping,
    probe_headers,
)


class FakeLlmProvider:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.temperatures: list[float] = []

    def complete(self, prompt: str, *, temperature: float, timeout: float) -> str:
        self.temperatures.append(temperature)
        if isinstance(self.payload, Exception):
            raise self.payload
        if isinstance(self.payload, dict):
            return json.dumps(self.payload)
        return str(self.payload)


_BASELINE_MAP = {
    "sku_id": "Item_ID",
    "timestamp": "Date",
    "demand_qty": "Avg_Usage_Per_Day",
    "lead_time_days": "Restock_Lead_Time",
}


def test_probe_baseline_y_temperatura_cero() -> None:
    marco = pd.DataFrame(
        {
            "Date": ["2024-10-01"],
            "Item_ID": ["105"],
            "Avg_Usage_Per_Day": ["108"],
            "Restock_Lead_Time": ["17"],
            "Current_Stock": ["1542"],
        }
    )
    fake = FakeLlmProvider(_BASELINE_MAP)
    artefacto = probe_headers(marco, fake, timeout=8.0)
    assert fake.temperatures == [0.0]
    assert list(artefacto.frame.columns) == [
        "sku_id",
        "timestamp",
        "demand_qty",
        "lead_time_days",
    ]


def test_csv_sin_senal_de_demanda_fail_closed() -> None:
    marco = pd.DataFrame({"Color": ["rojo"], "Foo": ["1"]})
    fake = FakeLlmProvider(
        {
            "sku_id": None,
            "timestamp": None,
            "demand_qty": None,
            "lead_time_days": None,
        }
    )
    with pytest.raises(SemanticAlignmentError, match="no se identificaron"):
        probe_headers(marco, fake)


def test_json_invalido_fail_closed() -> None:
    with pytest.raises(SemanticAlignmentError, match="JSON"):
        parse_header_mapping("esto no es json")


def test_timeout_se_propaga() -> None:
    marco = pd.DataFrame({"Date": ["2024-10-01"], "Item_ID": ["1"]})
    fake = FakeLlmProvider(LlmTimeoutError("El proveedor LLM no respondio en 1.0s"))
    with pytest.raises(LlmTimeoutError):
        probe_headers(marco, fake, timeout=1.0)
