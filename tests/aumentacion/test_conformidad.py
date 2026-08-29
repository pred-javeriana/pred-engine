"""0.4-A2 - Pruebas del validador de conformidad de esquema."""

from __future__ import annotations

import pandas as pd
import pytest

from pred_engine.aumentacion.conformidad import (
    validar_conformidad_o_fallar,
    verificar_conformidad,
)
from pred_engine.aumentacion.errores import SchemaConformanceError


def _marco_conforme() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": pd.array(["A", "B"], dtype="string"),
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "demand_qty": pd.array([3, 0], dtype="int64"),
            "lead_time_days": pd.array([2, 5], dtype="int64"),
        }
    )


def test_marco_conforme_pasa_y_genera_reporte() -> None:
    reporte = validar_conformidad_o_fallar(_marco_conforme())
    assert reporte.conforme
    assert reporte.row_count == 2
    assert reporte.fallas == ()


def test_columna_sobrante_aborta_la_exportacion() -> None:
    marco = _marco_conforme()
    marco["extra"] = 1
    with pytest.raises(SchemaConformanceError):
        validar_conformidad_o_fallar(marco)


def test_orden_de_columnas_incorrecto_es_no_conforme() -> None:
    marco = _marco_conforme()[["timestamp", "sku_id", "demand_qty", "lead_time_days"]]
    reporte = verificar_conformidad(marco)
    assert not reporte.conforme
    assert any("orden_canonico" in f for f in reporte.fallas)


def test_demanda_flotante_es_no_conforme() -> None:
    marco = _marco_conforme()
    marco["demand_qty"] = marco["demand_qty"].astype("float64")
    reporte = verificar_conformidad(marco)
    assert not reporte.conforme


def test_reporte_se_genera_incluso_cuando_falla() -> None:
    marco = _marco_conforme()
    marco["lead_time_days"] = pd.array([0, 5], dtype="int64")
    reporte = verificar_conformidad(marco)
    assert not reporte.conforme
    assert reporte.row_count == 2
    assert any("lead_time" in f for f in reporte.fallas)
