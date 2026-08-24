"""Pruebas del exportador Parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pred_engine.ingesta.data import RawWritePermissionError, ensure_data_layout
from pred_engine.ingesta.lector import export_parquet, extract_csv


def test_escribe_parquet_conservando_columnas(tmp_path: Path) -> None:
    layout = ensure_data_layout(tmp_path / "data")
    marco = pd.DataFrame({"sku id": ["A"], "qty": ["1"]})
    destino = layout.processed / "ventas.parquet"
    escrito = export_parquet(marco, destino, data_root=layout.root)
    leido = pd.read_parquet(escrito, engine="pyarrow")
    assert list(leido.columns) == ["sku id", "qty"]
    assert leido.iloc[0]["sku id"] == "A"


def test_rechaza_destino_en_raw(tmp_path: Path) -> None:
    layout = ensure_data_layout(tmp_path / "data")
    marco = pd.DataFrame({"sku": ["A"]})
    destino = layout.raw / "no.parquet"
    with pytest.raises(RawWritePermissionError):
        export_parquet(marco, destino, data_root=layout.root)


def test_rechaza_extension_incorrecta(tmp_path: Path) -> None:
    layout = ensure_data_layout(tmp_path / "data")
    marco = pd.DataFrame({"sku": ["A"]})
    with pytest.raises(ValueError, match=".parquet"):
        export_parquet(marco, layout.processed / "out.csv", data_root=layout.root)


def test_pipeline_csv_a_parquet(tmp_path: Path) -> None:
    layout = ensure_data_layout(tmp_path / "data")
    fuente = layout.raw / "ventas.csv"
    fuente.write_text("sku,qty\nA,3\nB,0\n", encoding="utf-8")
    artefacto = extract_csv(fuente, data_root=layout.root)
    destino = layout.processed / "ventas.parquet"
    export_parquet(artefacto.frame, destino, data_root=layout.root)
    leido = pd.read_parquet(destino, engine="pyarrow")
    assert len(leido) == 2
    assert list(leido.columns) == ["sku", "qty"]
