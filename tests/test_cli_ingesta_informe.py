"""CLI de ingesta: informe en archivo y conflicto de nombre en raw/ (G1, G5)."""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from pred_engine import cli


def _csv(ruta: Path, salto: int = 0) -> Path:
    inicio = date(2026, 1, 1)
    filas = ["sku_id,timestamp,demand_qty,lead_time_days"]
    for i in range(60):
        dia = (inicio + timedelta(days=i)).isoformat()
        filas.append(f"FIL-0001,{dia},{5 + i % 7 + salto},7")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(filas) + "\n", encoding="utf-8")
    return ruta


def _sha(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def test_classify_deja_el_informe_y_lo_imprime(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv = _csv(tmp_path / "e" / "ventas.csv")
    raiz = tmp_path / "data"

    codigo = cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz)])

    informe = raiz / "ingestas" / f"{_sha(csv)}.json"
    assert codigo == 0
    assert f"informe: {informe}" in capsys.readouterr().out
    assert json.loads(informe.read_text(encoding="utf-8"))["status"] == "accepted"


def test_classify_sale_con_7_ante_un_conflicto_y_no_pisa_nada(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raiz = tmp_path / "data"
    primero = _csv(tmp_path / "a" / "ventas.csv")
    assert cli.main(["classify", "--csv", str(primero), "--data-root", str(raiz)]) == 0
    guardado = (raiz / "raw" / "ventas.csv").read_bytes()
    nuevo = _csv(tmp_path / "b" / "ventas.csv", salto=3)
    capsys.readouterr()

    codigo = cli.main(["classify", "--csv", str(nuevo), "--data-root", str(raiz)])

    assert codigo == 7
    assert "confirme para conservar ambos" in capsys.readouterr().err
    assert (raiz / "raw" / "ventas.csv").read_bytes() == guardado
    informe = json.loads(
        (raiz / "ingestas" / f"{_sha(nuevo)}.json").read_text(encoding="utf-8")
    )
    assert informe["status"] == "needs_confirmation"


def test_classify_con_confirmacion_conserva_ambos(tmp_path: Path) -> None:
    raiz = tmp_path / "data"
    primero = _csv(tmp_path / "a" / "ventas.csv")
    assert cli.main(["classify", "--csv", str(primero), "--data-root", str(raiz)]) == 0
    nuevo = _csv(tmp_path / "b" / "ventas.csv", salto=3)

    codigo = cli.main(
        [
            "classify",
            "--csv",
            str(nuevo),
            "--data-root",
            str(raiz),
            "--confirm-overwrite",
        ]
    )

    assert codigo == 0
    assert len(list((raiz / "raw").iterdir())) == 2
    informe = json.loads(
        (raiz / "ingestas" / f"{_sha(nuevo)}.json").read_text(encoding="utf-8")
    )
    assert informe["status"] == "accepted"
    assert informe["deposit"]["would_overwrite"] is True


def test_classify_con_una_fila_invalida_sale_con_3_y_deja_el_informe_fallido(
    tmp_path: Path,
) -> None:
    csv = _csv(tmp_path / "e" / "ventas.csv")
    with csv.open("a", encoding="utf-8") as archivo:
        archivo.write("FIL-0002,2026-01-01,5,abc\n")
    raiz = tmp_path / "data"

    codigo = cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz)])

    assert codigo == 3
    informe = json.loads(
        (raiz / "ingestas" / f"{_sha(csv)}.json").read_text(encoding="utf-8")
    )
    assert informe["status"] == "failed"
    assert informe["error"]["code"] == "ingest.schema_barrier"
