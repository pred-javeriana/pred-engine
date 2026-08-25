"""Sonda consultiva con proveedor falso: aceptado, rechazado, timeout, JSON basura."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pred_engine.comun.llm import LlmTimeoutError
from pred_engine.ingesta.sonda import (
    SemanticAlignmentError,
    parse_header_diagnostic,
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


def _marco_baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2024-10-01"],
            "Item_ID": ["105"],
            "Avg_Usage_Per_Day": ["108"],
            "Restock_Lead_Time": ["17"],
            "Current_Stock": ["1542"],
        }
    )


def _reporte_rechazo_baseline() -> dict[str, object]:
    return {
        "status": "rejected",
        "diagnostic": [
            {
                "field": "timestamp",
                "severity": "error",
                "message": "La columna 'Date' no coincide con el contrato",
                "action": "Renombrar la columna 'Date' a 'timestamp'",
            },
            {
                "field": "sku_id",
                "severity": "error",
                "message": "La columna 'Item_ID' no coincide con el contrato",
                "action": "Renombrar la columna 'Item_ID' a 'sku_id'",
            },
        ],
    }


def test_probe_aceptado_no_muta_el_marco() -> None:
    marco = pd.DataFrame(
        {
            "sku_id": ["105"],
            "timestamp": ["2024-10-01"],
            "demand_qty": ["108"],
            "lead_time_days": ["17"],
        }
    )
    columnas_antes = list(marco.columns)
    fake = FakeLlmProvider(
        {
            "status": "accepted",
            "diagnostic": [
                {
                    "field": "schema",
                    "severity": "info",
                    "message": "Las cabeceras cumplen el contrato PRED",
                    "action": None,
                }
            ],
        }
    )
    artefacto = probe_headers(marco, fake, timeout=8.0)
    assert fake.temperatures == [0.0]
    assert list(artefacto.frame.columns) == columnas_antes
    assert artefacto.diagnostic.is_accepted()


def test_probe_rechazado_fail_fast_con_json() -> None:
    marco = _marco_baseline()
    fake = FakeLlmProvider(_reporte_rechazo_baseline())
    with pytest.raises(SemanticAlignmentError, match="rechazo las cabeceras") as exc:
        probe_headers(marco, fake)
    assert exc.value.diagnostic is not None
    assert exc.value.diagnostic.is_rejected()
    assert list(marco.columns) == [
        "Date",
        "Item_ID",
        "Avg_Usage_Per_Day",
        "Restock_Lead_Time",
        "Current_Stock",
    ]


def test_probe_rechazado_serializa_json() -> None:
    marco = _marco_baseline()
    fake = FakeLlmProvider(_reporte_rechazo_baseline())
    with pytest.raises(SemanticAlignmentError) as exc:
        probe_headers(marco, fake)
    carga = json.loads(exc.value.diagnostic_json())
    assert carga["status"] == "rejected"
    assert len(carga["diagnostic"]) >= 1
    assert "Renombrar" in carga["diagnostic"][0]["action"]


def test_csv_sin_senal_de_demanda_rechazado() -> None:
    marco = pd.DataFrame({"Color": ["rojo"], "Foo": ["1"]})
    fake = FakeLlmProvider(
        {
            "status": "rejected",
            "diagnostic": [
                {
                    "field": "schema",
                    "severity": "error",
                    "message": "No es un dataset de demanda/inventario",
                    "action": (
                        "Provea un CSV con sku_id, timestamp, "
                        "demand_qty, lead_time_days"
                    ),
                }
            ],
        }
    )
    with pytest.raises(SemanticAlignmentError):
        probe_headers(marco, fake)


def test_json_invalido_fail_closed() -> None:
    with pytest.raises(SemanticAlignmentError, match="JSON"):
        parse_header_diagnostic("esto no es json")


def test_json_sin_status_fail_closed() -> None:
    with pytest.raises(SemanticAlignmentError, match="contrato"):
        parse_header_diagnostic('{"diagnostic": []}')


def test_timeout_se_propaga() -> None:
    marco = pd.DataFrame({"Date": ["2024-10-01"], "Item_ID": ["1"]})
    fake = FakeLlmProvider(LlmTimeoutError("El proveedor LLM no respondio en 1.0s"))
    with pytest.raises(LlmTimeoutError):
        probe_headers(marco, fake, timeout=1.0)


def test_parse_header_diagnostic_acepta_lista_vacia() -> None:
    reporte = parse_header_diagnostic('{"status": "accepted", "diagnostic": []}')
    assert reporte.is_accepted()
    assert reporte.diagnostic == ()
