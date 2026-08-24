"""Pruebas del contrato Pydantic de observacion y mapeo."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from pred_engine.comun.modelos import (
    CANONICAL_FIELDS,
    HeaderMapping,
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


def test_mapeo_baseline_inventory_data() -> None:
    mapeo = HeaderMapping(
        sku_id="Item_ID",
        timestamp="Date",
        demand_qty="Avg_Usage_Per_Day",
        lead_time_days="Restock_Lead_Time",
    )
    assert mapeo.unmapped_fields() == ()
    assert mapeo.source_to_canonical()["Item_ID"] == "sku_id"


def test_mapeo_incompleto_lista_faltantes() -> None:
    mapeo = HeaderMapping(sku_id="Item_ID", timestamp="Date")
    assert set(mapeo.unmapped_fields()) == {"demand_qty", "lead_time_days"}


def test_campos_canonico_estables() -> None:
    assert CANONICAL_FIELDS == (
        "sku_id",
        "timestamp",
        "demand_qty",
        "lead_time_days",
    )
