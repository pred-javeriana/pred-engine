"""Pruebas del contrato Pydantic de observacion y diagnostico."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from pred_engine.comun.modelos import (
    CANONICAL_FIELDS,
    DiagnosticEntry,
    HeaderDiagnostic,
    InventoryObservation,
)


def test_observacion_valida_del_dataset_canonico() -> None:
    fila = InventoryObservation(
        sku_id="105",
        timestamp=datetime(2024, 10, 1),
        demand_qty=108.0,
        lead_time_days=17,
    )
    assert fila.sku_id == "105"
    assert fila.demand_qty >= 0
    assert fila.lead_time_days >= 1


def test_rechaza_demanda_negativa() -> None:
    with pytest.raises(ValidationError):
        InventoryObservation(
            sku_id="105",
            timestamp=datetime(2024, 10, 1),
            demand_qty=-1.0,
            lead_time_days=17,
        )


def test_rechaza_lead_time_cero() -> None:
    with pytest.raises(ValidationError):
        InventoryObservation(
            sku_id="105",
            timestamp=datetime(2024, 10, 1),
            demand_qty=10.0,
            lead_time_days=0,
        )


def test_strict_no_acepta_string_en_demanda() -> None:
    with pytest.raises(ValidationError):
        InventoryObservation(
            sku_id="105",
            timestamp=datetime(2024, 10, 1),
            demand_qty="108",  # type: ignore[arg-type]
            lead_time_days=17,
        )


def test_diagnostico_aceptado_vacio() -> None:
    reporte = HeaderDiagnostic(status="accepted", diagnostic=())
    assert reporte.is_accepted()
    assert not reporte.is_rejected()


def test_diagnostico_rechazado_con_entradas() -> None:
    reporte = HeaderDiagnostic(
        status="rejected",
        diagnostic=(
            DiagnosticEntry(
                field="timestamp",
                message="Falta columna timestamp",
                action="Renombrar 'Date' a 'timestamp'",
            ),
        ),
    )
    assert reporte.is_rejected()
    assert reporte.diagnostic[0].action == "Renombrar 'Date' a 'timestamp'"


def test_diagnostico_rechaza_status_invalido() -> None:
    with pytest.raises(ValidationError):
        HeaderDiagnostic(status="maybe", diagnostic=())  # type: ignore[arg-type]


def test_campos_canonico_estables() -> None:
    assert CANONICAL_FIELDS == (
        "sku_id",
        "timestamp",
        "demand_qty",
        "lead_time_days",
    )
