"""Reentrada Parquet con varios informes M1, antes del registro M3."""

import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from pred_engine.comun.errores_codigo import codigo_de
from pred_engine.forecasting.persistencia_evidencia import abrir_base_plataforma
from pred_engine.forecasting.persistencia_evidencia.referencias import origen_conocido
from pred_engine.run_artifacts import sha256_file
from tests._plataforma import crear_base_plataforma

pytestmark = pytest.mark.slow

# 400 dias: reserva de 80 con H = 2, asi los bloques de seleccion (31 ventanas)
# y de prueba (47) superan n_min y el veredicto puede ser EXPLORATORIO (ADR-023).
DIAS = 400


def ejecutar(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(Path(sys.executable).with_name("pred-engine")), *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize(
    "origenes", [(True, True), (None, None), (None, True), (True, None), (True, False)]
)
@pytest.mark.parametrize("ruta", ["publicada", "copia", "compartida"])
def test_run_parquet_con_informes_de_los_mismos_bytes(tmp_path, origenes, ruta):
    root = tmp_path / "data"
    db = crear_base_plataforma(tmp_path / "central.db")
    source = tmp_path / "a.csv"
    pd.DataFrame(
        {
            "sku_id": ["SKU"] * DIAS,
            "timestamp": pd.date_range("2024-01-01", periods=DIAS),
            "demand_qty": [20 + (i % 19) * 0.5 for i in range(DIAS)],
            "lead_time_days": 3,
        }
    ).to_csv(source, index=False)
    otra = tmp_path / "b.csv"
    otra.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    assert sha256_file(source) != sha256_file(otra)
    informes = []
    for csv in (source, otra):
        flags = [] if origenes[0] is None else ["--datos-sinteticos"]
        result = ejecutar(
            "classify", "--csv", str(csv), "--data-root", str(root), *flags
        )
        assert result.returncode == 0, result.stdout + result.stderr
        path = root / "ingestas" / f"{sha256_file(csv)}.json"
        report = json.loads(path.read_text())
        assert report["status"] == "accepted"
        informes.append((path, report))
    assert (
        informes[0][1]["published"]["sha256"] == informes[1][1]["published"]["sha256"]
    )
    parquet = root / informes[0][1]["published"]["parquet_path"]
    contenido = parquet.read_bytes()
    if origenes[0] != origenes[1]:
        # Estado historico contradictorio: no depende de que M1 permita crearlo hoy.
        path, report = informes[1]
        report["source_file"]["datos_sinteticos"] = origenes[1]
        path.write_text(json.dumps(report))
    if ruta == "compartida":
        path, report = informes[1]
        report["published"]["parquet_path"] = informes[0][1]["published"][
            "parquet_path"
        ]
        path.write_text(json.dumps(report))
    elif ruta == "copia":
        destino = tmp_path / "copia.parquet"
        shutil.copy2(parquet, destino)
        parquet = destino
    with abrir_base_plataforma(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM m3_ingestas").fetchone() == (0,)
    antes = db.read_bytes()
    result = ejecutar(
        "run",
        "--parquet",
        str(parquet),
        "--data-root",
        str(root),
        "--db-path",
        str(db),
        "--run-id",
        "compartida",
        "--families",
        "ml",
        "--trials",
        "1",
        "--min-train",
        "40",
        "--horizon",
        "2",
        "--step",
        "14",
        "--workers",
        "1",
        "--seed",
        "17",
        "--telemetry-interval",
        "0",
    )
    assert parquet.read_bytes() == contenido
    if set(origenes) == {True, False}:
        assert result.returncode == 1, result.stdout + result.stderr
        assert "origen" in result.stderr
        assert db.read_bytes() == antes
        assert not (root / "runs").exists()
        with abrir_base_plataforma(db) as conn, pytest.raises(ValueError) as error:
            origen_conocido(
                conn,
                fuente_sha256=None,
                parquet=parquet,
                data_root=root,
                ingesta_ref_m1=sha256_file(parquet),
            )
        assert codigo_de(error.value) == "provenance.origin_conflict"
        return
    assert result.returncode == 0, result.stdout + result.stderr
    declaradas = [csv for csv, o in zip((source, otra), origenes) if o is not None]
    if ruta == "publicada":
        esperado_sha = sha256_file(source)
    elif len(declaradas) == 1:
        esperado_sha = sha256_file(declaradas[0])
    else:
        esperado_sha = sha256_file(parquet)
    origen = next((valor for valor in origenes if valor is not None), None)
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT ingesta_sha256, estado, datos_sinteticos, origen_declarado "
            "FROM m3_corridas WHERE run_id_m2 = 'compartida'"
        ).fetchall() == [
            (
                esperado_sha,
                "PUBLICADA",
                int(origen is True),
                int(origen is not None),
            )
        ]
        veredictos = conn.execute(
            "SELECT veredicto, dm_p_valor FROM m3_veredictos"
        ).fetchall()
        assert veredictos == [("EXPLORATORIO", None)]
    corrida = json.loads((root / "runs/compartida/corrida.json").read_text())
    assert corrida["settings"]["datos_sinteticos"] is origen
