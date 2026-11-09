"""La corrida conserva las huellas de los bytes que realmente consumio."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from pred_engine import cli
from pred_engine.pipeline import Pipeline
from tests._plataforma import crear_base_plataforma


def _publicaciones(tmp_path: Path) -> tuple[Path, Path]:
    publicaciones = []
    for nombre, nivel in (("a", 20), ("b", 40)):
        csv = tmp_path / f"{nombre}.csv"
        pd.DataFrame(
            {
                "sku_id": ["sku"] * 120,
                "timestamp": pd.date_range("2024-01-01", periods=120),
                "demand_qty": [nivel + i % 7 for i in range(120)],
                "lead_time_days": [2] * 120,
            }
        ).to_csv(csv, index=False)
        raiz = tmp_path / nombre
        assert cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz)]) == 0
        publicaciones.append(raiz / "processed" / f"{nombre}.parquet")
    return publicaciones[0], publicaciones[1]


def _args(tmp_path: Path, fuente: Path, formato: str = "parquet") -> list[str]:
    db = crear_base_plataforma(tmp_path / "pred.db")
    return [
        "run",
        f"--{formato}",
        str(fuente),
        "--data-root",
        str(tmp_path / "run"),
        "--db-path",
        str(db),
        "--families",
        "ml",
        "dl",
        "--trials",
        "1",
        "--min-train",
        "40",
        "--horizon",
        "2",
        "--step",
        "7",
        "--workers",
        "1",
        "--telemetry-interval",
        "0",
        "--seed",
        "17",
    ]


@pytest.mark.parametrize("formato", ["csv", "parquet"])
def test_run_rechaza_fuente_sustituida_antes_de_l1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    formato: str,
) -> None:
    a, b = _publicaciones(tmp_path)
    if formato == "csv":
        a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    args = _args(tmp_path, a, formato)
    ingest = Pipeline.ingest

    def sustituir(self, request):
        a.write_bytes(b.read_bytes())
        return ingest(self, request)

    monkeypatch.setattr(Pipeline, "ingest", sustituir)
    codigo = cli.main(args)
    salida = capsys.readouterr()
    assert codigo == 1
    assert "el archivo cambio durante la corrida" in salida.out + salida.err
    corrida = json.loads(
        next((tmp_path / "run" / "runs").glob("*/corrida.json")).read_text()
    )
    assert corrida["summary"]["stages"][0]["state"] == "failed"
    with sqlite3.connect(tmp_path / "pred.db") as conn:
        assert conn.execute("SELECT count(*) FROM m3_corridas").fetchone()[0] == 0


@pytest.mark.parametrize("cambio", ["sustituir", "borrar"])
def test_run_conserva_huella_l1_al_persistir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cambio: str,
) -> None:
    a, b = _publicaciones(tmp_path)
    huella = hashlib.sha256(a.read_bytes()).hexdigest()
    args = _args(tmp_path, a)
    ejecutar = cli._run_pipeline

    def cambiar_despues_de_evaluar(pipeline, request):
        resultado = ejecutar(pipeline, request)
        if cambio == "sustituir":
            a.write_bytes(b.read_bytes())
        else:
            a.unlink()
        # La reconstruccion del manifiesto debe usar la misma publicacion de L1.
        next((tmp_path / "run" / "runs").glob("*/candidatos.json")).unlink()
        return resultado

    monkeypatch.setattr(cli, "_run_pipeline", cambiar_despues_de_evaluar)
    assert cli.main(args) in (0, 8)
    directorio = next((tmp_path / "run" / "runs").iterdir())
    corrida = json.loads((directorio / "corrida.json").read_text())
    assert corrida["inputs"]["source"]["sha256"] == huella
    assert corrida["inputs"]["m1"]["sha256"] == huella
    manifiesto = json.loads((directorio / "candidatos.json").read_text())
    assert manifiesto["contexto"]["ingesta_ref_m1"] == huella
    with sqlite3.connect(tmp_path / "pred.db") as conn:
        (estado,) = conn.execute("SELECT estado FROM m3_corridas").fetchone()
        (protocolo,) = conn.execute("SELECT manifiesto FROM m3_protocolos").fetchone()
    assert estado == "PUBLICADA"
    assert json.loads(protocolo)["contexto"]["ingesta_ref_m1"] == huella
