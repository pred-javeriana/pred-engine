"""Pruebas del extractor CSV pasivo y del hash SHA-256."""

from __future__ import annotations

from pathlib import Path

import pytest

from pred_engine.ingesta.data import ensure_data_layout
from pred_engine.ingesta.lector import extract_csv, hash_sha256_archivo


def _escribir_csv(ruta: Path, cuerpo: str) -> None:
    ruta.write_text(cuerpo, encoding="utf-8")


def test_lee_sin_inferir_fechas_ni_tipos(tmp_path: Path) -> None:
    layout = ensure_data_layout(tmp_path / "data")
    fuente = layout.raw / "ventas.csv"
    _escribir_csv(fuente, "fecha,qty\n2024-01-01,10\n2024-01-02,0\n")
    artefacto = extract_csv(fuente, data_root=layout.root)
    assert artefacto.frame.iloc[0]["fecha"] == "2024-01-01"
    assert artefacto.frame.iloc[0]["qty"] == "10"
    assert artefacto.row_count == 2
    assert artefacto.sha256 == hash_sha256_archivo(fuente)


def test_hash_estable_para_el_mismo_archivo(tmp_path: Path) -> None:
    layout = ensure_data_layout(tmp_path / "data")
    fuente = layout.raw / "ventas.csv"
    _escribir_csv(fuente, "sku,qty\nA,1\n")
    primero = extract_csv(fuente, data_root=layout.root)
    segundo = extract_csv(fuente, data_root=layout.root)
    assert primero.sha256 == segundo.sha256


def test_rechaza_rutas_fuera_de_raw(tmp_path: Path) -> None:
    layout = ensure_data_layout(tmp_path / "data")
    fuente = layout.staging / "no.csv"
    _escribir_csv(fuente, "sku,qty\nA,1\n")
    with pytest.raises(ValueError, match="solo lee"):
        extract_csv(fuente, data_root=layout.root)


def test_rechaza_no_csv(tmp_path: Path) -> None:
    layout = ensure_data_layout(tmp_path / "data")
    fuente = layout.raw / "nota.txt"
    fuente.write_text("hola", encoding="utf-8")
    with pytest.raises(ValueError, match=".csv"):
        extract_csv(fuente, data_root=layout.root)


def test_mas_de_cincuenta_mil_filas(tmp_path: Path) -> None:
    layout = ensure_data_layout(tmp_path / "data")
    fuente = layout.raw / "grande.csv"
    n_filas = 50_001
    lineas = ["sku,qty\n", *[f"A,{indice}\n" for indice in range(n_filas)]]
    fuente.write_text("".join(lineas), encoding="utf-8")
    artefacto = extract_csv(fuente, data_root=layout.root)
    assert artefacto.row_count == n_filas
    assert len(artefacto.sha256) == 64
