"""El lector rechaza metadatos incompletos sin inventar datos de auditoria."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from pred_engine import cli
from pred_engine.comun.metadatos_parquet import METADATA_KEY


@pytest.mark.parametrize("comando", ["verify", "classify"])
@pytest.mark.parametrize(
    ("campo", "valor", "eliminar"),
    [
        ("fraction", None, True),
        ("exclusions", None, True),
        ("fraction", None, False),
        ("fraction", "0.2", False),
        ("fraction", True, False),
        ("fraction", [], False),
        ("exclusions", None, False),
        ("exclusions", {}, False),
        ("exclusions", "", False),
        ("exclusions", [None], False),
        ("exclusions", [{"sku_id": "TARDIO"}], False),
        ("exclusions", [{"sku_id": 12, "cause": "tardio"}], False),
        ("exclusions", [{"sku_id": "TARDIO", "cause": None}], False),
    ],
)
def test_cli_rechaza_campos_obligatorios_ausentes_o_de_tipo_incorrecto(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    comando: str,
    campo: str,
    valor: object,
    eliminar: bool,
) -> None:
    csv = tmp_path / "panel.csv"
    pd.DataFrame(
        {
            "sku_id": ["A"] * 10,
            "timestamp": pd.date_range("2024-01-01", periods=10),
            "demand_qty": [5.0] * 10,
            "lead_time_days": [2] * 10,
        }
    ).to_csv(csv, index=False)
    raiz = tmp_path / "original"
    assert cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz)]) == 0
    tabla = pq.read_table(raiz / "processed" / "panel.parquet")
    metadatos = dict(tabla.schema.metadata or {})
    payload = json.loads(metadatos[METADATA_KEY])
    if eliminar:
        del payload[campo]
    else:
        payload[campo] = valor
    metadatos[METADATA_KEY] = json.dumps(payload).encode("utf-8")
    incompleto = tmp_path / "incompleto.parquet"
    pq.write_table(tabla.replace_schema_metadata(metadatos), incompleto)
    capsys.readouterr()

    salida = tmp_path / "salida"
    argumentos = [comando, "--parquet", str(incompleto)]
    if comando == "classify":
        argumentos += ["--data-root", str(salida)]
    assert cli.main(argumentos) == 6
    consola = capsys.readouterr()
    assert "re-import from the CSV" in consola.err
    assert "contrato_1_4: accepted" not in consola.out
    assert not (salida / "processed" / "incompleto.parquet").exists()
