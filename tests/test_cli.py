"""CLI: mapeo ok, fail-closed, catalogo de modelos y ausencia de clave en stdout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pred_engine import cli


class FakeLlmProvider:
    def complete(self, prompt: str, *, temperature: float, timeout: float) -> str:
        return json.dumps(
            {
                "sku_id": "Item_ID",
                "timestamp": "Date",
                "demand_qty": "Avg_Usage_Per_Day",
                "lead_time_days": "Restock_Lead_Time",
            }
        )


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


def test_cli_ingest_con_proveedor_inyectado(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv = tmp_path / "mini.csv"
    csv.write_text(
        "Date,Item_ID,Avg_Usage_Per_Day,Restock_Lead_Time\n2024-10-01,105,108,17\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "build_llm_provider",
        lambda *args, **kwargs: FakeLlmProvider(),
    )
    codigo = cli.main(
        [
            "ingest",
            "--csv",
            str(csv),
            "--provider",
            "gemini",
            "--api-key",
            "no-se-debe-imprimir",
            "--data-root",
            str(tmp_path / "data"),
        ]
    )
    assert codigo == 0
    salida = capsys.readouterr().out
    assert "no-se-debe-imprimir" not in salida
    assert "filas_panel_diario" in salida


def test_cli_sin_clave_falla(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRED_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    csv = tmp_path / "mini.csv"
    csv.write_text("Date,Item_ID\n2024-10-01,1\n", encoding="utf-8")
    codigo = cli.main(
        ["ingest", "--csv", str(csv), "--data-root", str(tmp_path / "data")]
    )
    assert codigo == 1
