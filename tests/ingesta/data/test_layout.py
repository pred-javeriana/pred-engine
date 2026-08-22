"""Pruebas del arbol de directorios de ingesta."""

from __future__ import annotations

from pathlib import Path

from pred_engine.ingesta.data import DataLayout, ensure_data_layout


def test_crea_raw_staging_processed(tmp_path: Path) -> None:
    raiz = tmp_path / "data"
    layout = ensure_data_layout(raiz)
    assert layout.raw.is_dir()
    assert layout.staging.is_dir()
    assert layout.processed.is_dir()
    assert isinstance(layout, DataLayout)


def test_idempotente_si_los_directorios_ya_existen(tmp_path: Path) -> None:
    raiz = tmp_path / "data"
    primero = ensure_data_layout(raiz)
    segundo = ensure_data_layout(raiz)
    assert primero == segundo


def test_sin_estado_global(tmp_path: Path) -> None:
    layout_a = ensure_data_layout(tmp_path / "a")
    layout_b = ensure_data_layout(tmp_path / "b")
    assert layout_a.root != layout_b.root
