"""0.4-A1 - Pruebas del contrato de datos del artefacto de salida."""

from __future__ import annotations

from pred_engine.aumentacion.contrato import (
    CONTRACT_VERSION,
    OUTPUT_COLUMNS,
    OUTPUT_CONTRACT,
    describir_contrato,
)


def test_contrato_declara_exactamente_cuatro_columnas() -> None:
    assert OUTPUT_COLUMNS == ("sku_id", "timestamp", "demand_qty", "lead_time_days")
    assert len(OUTPUT_CONTRACT) == 4


def test_contrato_tipa_demanda_y_lead_time_como_enteros() -> None:
    por_nombre = {c.name: c for c in OUTPUT_CONTRACT}
    assert por_nombre["demand_qty"].dtype == "int64"
    assert por_nombre["lead_time_days"].dtype == "int64"


def test_contrato_esta_versionado() -> None:
    assert CONTRACT_VERSION.count(".") == 2
    assert CONTRACT_VERSION in describir_contrato()
