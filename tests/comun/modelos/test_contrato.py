"""Pruebas del contrato Pydantic de observacion y diagnostico."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from pred_engine.comun.modelos import (
    ADI_THRESHOLD,
    CANONICAL_FIELDS,
    CV2_THRESHOLD,
    PANEL_DTYPES,
    PANEL_FIELDS,
    TOPOLOGY_FIELD,
    ClassifiedObservation,
    DiagnosticEntry,
    HeaderDiagnostic,
    InventoryObservation,
    TopologyMetrics,
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


def test_umbrales_syntetos_boylan() -> None:
    assert ADI_THRESHOLD == 1.32
    assert CV2_THRESHOLD == 0.49


def test_panel_conserva_canonico_y_anade_sku_class() -> None:
    assert TOPOLOGY_FIELD == "sku_class"
    assert PANEL_FIELDS == (
        "sku_id",
        "timestamp",
        "demand_qty",
        "lead_time_days",
        "sku_class",
    )


def test_observacion_clasificada_exige_sku_class() -> None:
    fila = ClassifiedObservation(
        sku_id="105",
        timestamp=datetime(2024, 10, 1),
        demand_qty=108.0,
        lead_time_days=17,
        sku_class="intermittent",
    )
    assert fila.sku_class == "intermittent"
    assert PANEL_DTYPES["sku_class"] == "string"


def test_observacion_clasificada_rechaza_etiqueta_invalida() -> None:
    with pytest.raises(ValidationError):
        ClassifiedObservation(
            sku_id="105",
            timestamp=datetime(2024, 10, 1),
            demand_qty=108.0,
            lead_time_days=17,
            sku_class="intermitente",  # type: ignore[arg-type]
        )


def test_metricas_topologia_validas() -> None:
    m = TopologyMetrics(
        sku_id="105",
        n_periods=4,
        n_positive=2,
        adi=2.0,
        cv2=0.25,
        sku_class="intermittent",
    )
    assert m.sku_class == "intermittent"


def test_metricas_rechazan_clase_en_espanol() -> None:
    with pytest.raises(ValidationError):
        TopologyMetrics(
            sku_id="105",
            n_periods=4,
            n_positive=2,
            adi=2.0,
            cv2=0.25,
            sku_class="intermitente",  # type: ignore[arg-type]
        )
