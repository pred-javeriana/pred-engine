"""Publicador 1.4: solo Parquet en processed/, tras validar."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from pred_engine.comun.modelos import PANEL_FIELDS
from pred_engine.ingesta.categorizacion import classify_daily_panel
from pred_engine.ingesta.data import RawWritePermissionError, ensure_data_layout
from pred_engine.ingesta.salida import (
    OutputContractError,
    PanelPreservationError,
    publish_classified_panel,
    read_classified_parquet,
)


def _diario_mini() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": pd.Series(["105"] * 4, dtype="string"),
            "timestamp": pd.date_range(datetime(2024, 10, 1), periods=4, freq="D"),
            "demand_qty": [108.0, 0.0, 0.0, 50.0],
            "lead_time_days": pd.Series([17, 17, 17, 17], dtype="int64"),
        }
    )


def test_publica_parquet_con_cinco_columnas(tmp_path: Path) -> None:
    layout = ensure_data_layout(tmp_path / "data")
    diario = _diario_mini()
    clasificado = classify_daily_panel(diario).frame
    destino = layout.processed / "panel.parquet"
    escrito = publish_classified_panel(
        clasificado,
        destino,
        data_root=layout.root,
        daily_panel=diario,
    )
    leido = pd.read_parquet(escrito, engine="pyarrow")
    assert list(leido.columns) == list(PANEL_FIELDS)
    assert len(leido) == 4
    assert escrito.suffix == ".parquet"
    assert not (layout.raw / "panel.parquet").exists()


def test_rechaza_csv_como_destino(tmp_path: Path) -> None:
    layout = ensure_data_layout(tmp_path / "data")
    diario = _diario_mini()
    clasificado = classify_daily_panel(diario).frame
    with pytest.raises(OutputContractError, match="parquet"):
        publish_classified_panel(
            clasificado,
            layout.processed / "panel.csv",
            data_root=layout.root,
            daily_panel=diario,
        )


def test_rechaza_escritura_en_raw(tmp_path: Path) -> None:
    layout = ensure_data_layout(tmp_path / "data")
    diario = _diario_mini()
    clasificado = classify_daily_panel(diario).frame
    with pytest.raises(RawWritePermissionError):
        publish_classified_panel(
            clasificado,
            layout.raw / "panel.parquet",
            data_root=layout.root,
            daily_panel=diario,
        )


def test_no_escribe_si_el_contrato_falla(tmp_path: Path) -> None:
    layout = ensure_data_layout(tmp_path / "data")
    diario = _diario_mini()
    clasificado = classify_daily_panel(diario).frame.copy()
    clasificado["sku_class"] = pd.Series(["nope"] * len(clasificado), dtype="string")
    destino = layout.processed / "panel.parquet"
    with pytest.raises(OutputContractError):
        publish_classified_panel(
            clasificado,
            destino,
            data_root=layout.root,
            daily_panel=diario,
        )
    assert not destino.exists()


def test_no_escribe_si_hay_reorden(tmp_path: Path) -> None:
    layout = ensure_data_layout(tmp_path / "data")
    diario = _diario_mini()
    clasificado = classify_daily_panel(diario).frame.iloc[::-1].reset_index(drop=True)
    destino = layout.processed / "panel.parquet"
    with pytest.raises(PanelPreservationError):
        publish_classified_panel(
            clasificado,
            destino,
            data_root=layout.root,
            daily_panel=diario,
        )
    assert not destino.exists()


def test_roundtrip_parquet_vuelve_a_validar(tmp_path: Path) -> None:
    layout = ensure_data_layout(tmp_path / "data")
    diario = _diario_mini()
    clasificado = classify_daily_panel(diario).frame
    destino = layout.processed / "panel.parquet"
    publish_classified_panel(
        clasificado,
        destino,
        data_root=layout.root,
        daily_panel=diario,
    )
    leido = read_classified_parquet(destino)
    assert list(leido.columns) == list(PANEL_FIELDS)
    assert set(leido["sku_class"].astype(str)) == {"intermittent"}
