"""Pruebas de la politica de solo lectura sobre data/raw."""

from __future__ import annotations

from pathlib import Path

import pytest

from pred_engine.ingesta.data import (
    RawWritePermissionError,
    enforce_raw_read_only,
    raw_read_only_guard,
)


def test_lectura_en_raw_permitida(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    objetivo = raw / "fuente.csv"
    objetivo.write_text("sku,qty\nA,1\n", encoding="utf-8")
    with raw_read_only_guard(raw), objetivo.open("r", encoding="utf-8") as fh:
        assert "sku" in fh.read()


def test_escritura_en_raw_lanza_permission_error(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    objetivo = raw / "fuente.csv"
    objetivo.write_text("ok\n", encoding="utf-8")
    with (
        raw_read_only_guard(raw),
        pytest.raises(RawWritePermissionError, match="fuente.csv"),
    ):
        objetivo.open("w", encoding="utf-8")


@pytest.mark.parametrize("modo", ["w", "a", "x", "w+", "r+", "wb", "ab"])
def test_modos_de_escritura_bloqueados(tmp_path: Path, modo: str) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    objetivo = raw / "fuente.csv"
    if "x" not in modo:
        objetivo.write_text("ok\n", encoding="utf-8")
    with raw_read_only_guard(raw), pytest.raises(RawWritePermissionError):
        open(objetivo, modo)  # noqa: SIM115


def test_escritura_fuera_de_raw_permitida(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    destino = tmp_path / "processed" / "out.txt"
    destino.parent.mkdir()
    with raw_read_only_guard(raw), destino.open("w", encoding="utf-8") as fh:
        fh.write("ok")
    assert destino.read_text(encoding="utf-8") == "ok"


def test_error_incluye_llamador(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    objetivo = raw / "fuente.csv"
    objetivo.write_text("ok\n", encoding="utf-8")

    def mutar_crudo() -> None:
        with raw_read_only_guard(raw), objetivo.open("w", encoding="utf-8"):
            pass

    with pytest.raises(RawWritePermissionError, match="mutar_crudo"):
        mutar_crudo()


def test_decorador_bloquea_escritura(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    objetivo = raw / "fuente.csv"
    objetivo.write_text("ok\n", encoding="utf-8")

    @enforce_raw_read_only(raw)
    def mutar() -> None:
        with objetivo.open("a", encoding="utf-8") as fh:
            fh.write("no")

    with pytest.raises(RawWritePermissionError):
        mutar()


def test_guardia_restaura_open(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    objetivo = raw / "fuente.csv"
    with raw_read_only_guard(raw):
        pass
    objetivo.write_text("despues\n", encoding="utf-8")
    assert objetivo.read_text(encoding="utf-8") == "despues\n"
