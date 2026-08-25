"""CLI: diagnostico ok, fail-closed, catalogo y clave ausente en stdout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pred_engine import cli
from pred_engine.comun.llm import LlmProviderError


class FakeLlmProvider:
    def complete(self, prompt: str, *, temperature: float, timeout: float) -> str:
        return json.dumps(
            {
                "status": "accepted",
                "diagnostic": [
                    {
                        "field": "schema",
                        "severity": "info",
                        "message": "Cabeceras canonicas presentes",
                        "action": None,
                    }
                ],
            }
        )


class Http503LlmProvider:
    def complete(self, prompt: str, *, temperature: float, timeout: float) -> str:
        raise LlmProviderError(
            "El proveedor LLM respondio HTTP 503",
            status_code=503,
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
        "sku_id,timestamp,demand_qty,lead_time_days\n105,2024-10-01,108,17\n",
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
            "test-key",
            "--data-root",
            str(tmp_path / "data"),
        ]
    )
    assert codigo == 0
    salida = capsys.readouterr().out
    assert "no-se-debe-imprimir" not in salida
    assert "filas_panel_diario" in salida
    assert "diagnostico:" in salida


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


def test_cli_error_http_llm_sale_limpio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv = tmp_path / "mini.csv"
    csv.write_text(
        "sku_id,timestamp,demand_qty,lead_time_days\n105,2024-10-01,108,17\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "build_llm_provider",
        lambda *args, **kwargs: Http503LlmProvider(),
    )
    codigo = cli.main(
        [
            "ingest",
            "--csv",
            str(csv),
            "--provider",
            "gemini",
            "--api-key",
            "test-key",
            "--data-root",
            str(tmp_path / "data"),
        ]
    )
    assert codigo == 1
    err = capsys.readouterr().err
    assert "503" in err
    assert "Traceback" not in err
