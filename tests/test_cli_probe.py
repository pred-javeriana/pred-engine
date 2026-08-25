"""CLI probe: diagnostico JSON en stderr al rechazar."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pred_engine.cli import main


def test_probe_rechazado_imprime_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv = tmp_path / "bad.csv"
    csv.write_text("Date,Item_ID\n2024-01-01,1\n", encoding="utf-8")
    raiz = tmp_path / "data"

    class FakeProvider:
        def complete(self, prompt: str, *, temperature: float, timeout: float) -> str:
            return json.dumps(
                {
                    "status": "rejected",
                    "diagnostic": [
                        {
                            "field": "timestamp",
                            "severity": "error",
                            "message": "Falta timestamp",
                            "action": "Renombrar Date a timestamp",
                        }
                    ],
                }
            )

    monkeypatch.setattr(
        "pred_engine.cli._build_provider",
        lambda args: ("gemini", "fake", FakeProvider()),
    )

    codigo = main(
        [
            "probe",
            "--csv",
            str(csv),
            "--data-root",
            str(raiz),
            "--provider",
            "gemini",
            "--api-key",
            "test-key",
        ]
    )
    assert codigo == 2
