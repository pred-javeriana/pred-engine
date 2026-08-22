"""CLI: catalogo de modelos y validacion de proveedor."""

from __future__ import annotations

from pathlib import Path

import pytest

from pred_engine import cli


def test_models_lista_gemini_3_5_flash(capsys: pytest.CaptureFixture[str]) -> None:
    codigo = cli.main(["models", "--provider", "gemini"])
    assert codigo == 0
    salida = capsys.readouterr().out
    assert "gemini-3.5-flash" in salida
    assert "gemini-2.5-flash-lite" in salida
    assert "default" in salida.lower() or "Default" in salida


def test_models_proveedor_invalido() -> None:
    assert cli.main(["models", "--provider", "cohere"]) == 1


def test_ingest_rechaza_modelo_fuera_de_catalogo(tmp_path: Path) -> None:
    csv = tmp_path / "mini.csv"
    csv.write_text("Date,Item_ID\n2024-10-01,1\n", encoding="utf-8")
    codigo = cli.main(
        [
            "ingest",
            "--csv",
            str(csv),
            "--provider",
            "gemini",
            "--api-key",
            "test-key",
            "--model",
            "gemini-ultra-falso",
            "--data-root",
            str(tmp_path / "data"),
        ]
    )
    assert codigo == 1
