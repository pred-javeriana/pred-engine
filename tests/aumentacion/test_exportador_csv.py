"""0.4-B2 - Pruebas del exportador CSV del artefacto final."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pred_engine.aumentacion.errores import SchemaConformanceError, WormOverwriteError
from pred_engine.aumentacion.exportador_csv import exportar_artefacto_csv


def _panel(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": pd.array([f"SKU{i % 3}" for i in range(n)], dtype="string"),
            "timestamp": pd.to_datetime("2024-01-01")
            + pd.to_timedelta(range(n), unit="D"),
            "demand_qty": pd.array([i % 7 for i in range(n)], dtype="int64"),
            "lead_time_days": pd.array([1 + i % 4 for i in range(n)], dtype="int64"),
        }
    )


def test_exporta_csv_en_orden_canonico_con_hash_y_conteo(tmp_path: Path) -> None:
    artefacto = exportar_artefacto_csv(
        _panel(10), "panel.csv", data_root=tmp_path / "data", minimo_filas=0
    )
    assert artefacto.row_count == 10
    assert len(artefacto.sha256) == 64
    leido = pd.read_csv(artefacto.path)
    assert list(leido.columns) == [
        "sku_id",
        "timestamp",
        "demand_qty",
        "lead_time_days",
    ]
    assert leido.iloc[0]["timestamp"] == "2024-01-01"


def test_umbral_de_cincuenta_mil_filas_se_exige_por_defecto(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="50000|50_000|filas"):
        exportar_artefacto_csv(_panel(10), "panel.csv", data_root=tmp_path / "data")


def test_artefacto_no_conforme_no_se_escribe(tmp_path: Path) -> None:
    panel = _panel(5)
    panel["demand_qty"] = panel["demand_qty"].astype("float64")
    with pytest.raises(SchemaConformanceError):
        exportar_artefacto_csv(
            panel, "panel.csv", data_root=tmp_path / "data", minimo_filas=0
        )
    assert not (tmp_path / "data" / "raw" / "panel.csv").exists()


def test_no_sobrescribe_un_artefacto_existente(tmp_path: Path) -> None:
    exportar_artefacto_csv(
        _panel(10), "panel.csv", data_root=tmp_path / "data", minimo_filas=0
    )
    with pytest.raises(WormOverwriteError):
        exportar_artefacto_csv(
            _panel(12), "panel.csv", data_root=tmp_path / "data", minimo_filas=0
        )
