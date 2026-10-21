"""La corrida conserva la topologia de la ingesta y respeta los conflictos (G1, G5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pred_engine.cli import main
from pred_engine.verify import representative_csv
from tests._plataforma import crear_base_plataforma

_ARGS = [
    "--trials",
    "1",
    "--min-train",
    "40",
    "--horizon",
    "2",
    "--step",
    "14",
    "--metric",
    "mae",
    "--seed",
    "17",
]


def _correr(
    csv: Path, raiz: Path, capsys: pytest.CaptureFixture[str], *extra: str
) -> tuple[int, dict[str, Any]]:
    if not (raiz / "pred.db").exists():
        crear_base_plataforma(raiz / "pred.db")
    codigo = main(["run", "--csv", str(csv), "--data-root", str(raiz), *_ARGS, *extra])
    ultima = capsys.readouterr().out.strip().splitlines()[-1]
    return codigo, json.loads(ultima)


def test_la_corrida_guarda_la_topologia_de_la_ingesta(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv = representative_csv(tmp_path / "source.csv")
    raiz = tmp_path / "data"

    _, resumen = _correr(csv, raiz, capsys)

    carpeta = Path(resumen["run_dir"])
    assert "topologia.json" in resumen["files"]
    topologia = json.loads((carpeta / "topologia.json").read_text(encoding="utf-8"))
    corrida = json.loads((carpeta / "corrida.json").read_text(encoding="utf-8"))
    assert topologia["schema"] == "pred-engine.topologia/1"
    assert topologia["thresholds"] == {"adi": 1.32, "cv2": 0.49}
    assert corrida["files"]["topologia"] == "topologia.json"

    informes = list((raiz / "ingestas").glob("*.json"))
    assert len(informes) == 1
    informe = json.loads(informes[0].read_text(encoding="utf-8"))
    assert topologia["source_sha256"] == informe["ingest_id"]
    assert topologia["metrics"] == informe["topology"]["metrics"]
    assert topologia["t_star"] == informe["reserve"]["t_star"]
    assert {m["sku_id"] for m in topologia["metrics"]}


def test_la_corrida_no_pisa_un_archivo_distinto_con_el_mismo_nombre(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raiz = tmp_path / "data"
    (tmp_path / "a").mkdir()
    primero = representative_csv(tmp_path / "a" / "source.csv")
    _correr(primero, raiz, capsys)
    guardado = (raiz / "raw" / "source.csv").read_bytes()
    otro = tmp_path / "b" / "source.csv"
    otro.parent.mkdir()
    texto = primero.read_text(encoding="utf-8").replace("18", "19", 1)
    otro.write_text(texto, encoding="utf-8")
    assert otro.read_bytes() != primero.read_bytes()

    codigo, resumen = _correr(otro, raiz, capsys)

    assert codigo == 1
    assert resumen.get("failed_stage") == "L1"
    assert (raiz / "raw" / "source.csv").read_bytes() == guardado
