"""`pred-engine origen`: el motor es el unico dueno del origen de una ingesta.

Escenarios que la plataforma resolvia por su cuenta y ahora pregunta al motor
antes de encolar: declaracion de M1, bytes publicados compartidos, ausencia,
configuraciones de todas las ejecuciones y corridas de origen desconocido.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pred_engine import cli
from pred_engine.run_artifacts import sha256_file
from tests._plataforma import crear_base_plataforma
from tests.forecasting.test_quinta_revision_m3 import correr
from tests.forecasting.test_tercera_revision_m3 import fuente

pytestmark = pytest.mark.slow


def _origen(capsys, root: Path, db: Path, sha: str, *declaracion: str):
    capsys.readouterr()
    codigo = cli.main(
        [
            "origen",
            "--ingest-id",
            sha,
            *declaracion,
            "--data-root",
            str(root),
            "--db-path",
            str(db),
        ]
    )
    salida = capsys.readouterr().out.strip().splitlines()
    assert len(salida) == 1, salida
    return codigo, json.loads(salida[0])


def _classify(root: Path, csv: Path, *declaracion: str) -> int:
    return cli.main(
        ["classify", "--csv", str(csv), "--data-root", str(root), *declaracion]
    )


def _informe(root: Path, csv: Path) -> dict:
    return json.loads((root / "ingestas" / f"{sha256_file(csv)}.json").read_text())


def _corridas(db: Path) -> list[tuple]:
    with sqlite3.connect(db) as conn:
        return conn.execute(
            "SELECT run_id_m2, datos_sinteticos, origen_declarado FROM m3_corridas "
            "ORDER BY run_id_m2"
        ).fetchall()


def _conflicto(resultado) -> None:
    codigo, cuerpo = resultado
    assert codigo == 3
    assert cuerpo["code"] == "provenance.origin_conflict"
    assert "no puede cambiar" in cuerpo["detail"]


def test_declaracion_de_m1_se_fija_y_viaja_a_la_corrida(tmp_path, capsys):
    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "pred.db")
    csv = fuente(tmp_path)
    sha = sha256_file(csv)
    assert _classify(root, csv, "--datos-sinteticos") == 0

    assert _origen(capsys, root, db, sha) == (
        0,
        {"ingest_id": sha, "datos_sinteticos": True},
    )
    assert _origen(capsys, root, db, sha, "--datos-sinteticos")[1]["datos_sinteticos"]
    _conflicto(_origen(capsys, root, db, sha, "--datos-reales"))
    # La corrida sin declaracion recibe el origen de M1.
    assert correr(db, root, "heredada", "--csv", str(csv)) == 0
    assert _corridas(db) == [("heredada", 1, 1)]
    # Volver a clasificar el mismo CSV como real se rechaza sin cambiar su informe.
    assert _classify(root, csv, "--datos-reales") == 1
    assert _informe(root, csv)["source_file"]["datos_sinteticos"] is True


def test_bytes_publicados_compartidos_heredan_y_no_admiten_el_opuesto(tmp_path, capsys):
    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "pred.db")
    sintetico = fuente(tmp_path).rename(tmp_path / "a.csv")
    # El mismo panel con otro fin de linea: otro CSV, los mismos bytes publicados.
    desconocido, real = tmp_path / "c.csv", tmp_path / "d.csv"
    desconocido.write_bytes(sintetico.read_bytes().replace(b"\n", b"\r\n"))
    real.write_bytes(sintetico.read_bytes() + b"\n")
    assert _classify(root, sintetico, "--datos-sinteticos") == 0
    assert _classify(root, desconocido) == 0
    huella = _informe(root, sintetico)["published"]["sha256"]
    assert _informe(root, desconocido)["published"]["sha256"] == huella
    assert _informe(root, desconocido)["source_file"]["datos_sinteticos"] is None

    sha = sha256_file(desconocido)
    assert _origen(capsys, root, db, sha) == (
        0,
        {"ingest_id": sha, "datos_sinteticos": True},
    )
    _conflicto(_origen(capsys, root, db, sha, "--datos-reales"))

    # Un informe que declara reales esos mismos bytes queda rechazado.
    capsys.readouterr()
    assert _classify(root, real, "--datos-reales") == 1
    assert "no puede cambiar" in capsys.readouterr().err
    rechazado = _informe(root, real)
    assert rechazado["status"] == "rejected"
    assert rechazado["error"]["code"] == "provenance.origin_conflict"
    assert rechazado["source_file"]["datos_sinteticos"] is False
    # Un informe rechazado no declara el origen de los bytes que comparte.
    assert _origen(capsys, root, db, sha)[1]["datos_sinteticos"] is True
    # Un Parquet sin registro en la base se identifica por los informes M1: la
    # ingesta sin declaracion no compite con la que declaro su origen.
    copia = tmp_path / "copia.parquet"
    copia.write_bytes(
        (root / _informe(root, sintetico)["published"]["parquet_path"]).read_bytes()
    )
    assert correr(db, root, "parquet", "--parquet", str(copia)) == 0
    assert correr(db, root, "compartida", "--csv", str(desconocido)) == 0
    assert _corridas(db) == [("compartida", 1, 1), ("parquet", 1, 1)]
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT ingesta_sha256 FROM m3_corridas WHERE run_id_m2 = 'parquet'"
        ).fetchone() == (sha256_file(sintetico),)


def test_la_ausencia_de_declaracion_nunca_es_dato_real(tmp_path, capsys):
    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "pred.db")
    csv = fuente(tmp_path)
    sha = sha256_file(csv)
    assert _classify(root, csv) == 0
    assert _origen(capsys, root, db, sha) == (
        0,
        {"ingest_id": sha, "datos_sinteticos": None},
    )
    assert correr(db, root, "desconocida", "--csv", str(csv)) == 0
    assert _corridas(db) == [("desconocida", 0, 0)]
    # Una corrida de origen desconocido no declara nada.
    assert _origen(capsys, root, db, sha)[1]["datos_sinteticos"] is None
    assert _origen(capsys, root, db, sha, "--datos-reales") == (
        0,
        {"ingest_id": sha, "datos_sinteticos": False},
    )
    # Declarar despues el origen no reescribe la corrida terminada sin declaracion.
    assert correr(db, root, "declarada", "--csv", str(csv), "--datos-reales") == 0
    assert _corridas(db) == [("declarada", 0, 1), ("desconocida", 0, 0)]
    with sqlite3.connect(db) as conn:
        ((parametros,),) = conn.execute(
            "SELECT c.parametros FROM configuraciones c JOIN ejecuciones e "
            "ON e.configuracion_id = c.id WHERE e.run_id = 'desconocida'"
        ).fetchall()
    assert json.loads(parametros)["datos_sinteticos"] is None
    _conflicto(_origen(capsys, root, db, sha, "--datos-sinteticos"))


def test_las_configuraciones_de_todas_las_ejecuciones_declaran_un_solo_origen(
    tmp_path, capsys
):
    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "pred.db")
    csv = fuente(tmp_path)
    sha = sha256_file(csv)
    assert _classify(root, csv) == 0
    with sqlite3.connect(db) as conn:
        ingesta = conn.execute(
            "INSERT INTO ingestas (sha256, nombre_archivo, filas, skus) "
            "VALUES (?, 'source.csv', 220, 1)",
            (sha,),
        ).lastrowid
        # Una ejecucion que la plataforma encolo como real y nunca corrio.
        configuracion = conn.execute(
            "INSERT INTO configuraciones (version, parametros) VALUES (1, ?)",
            (json.dumps({"datos_sinteticos": False, "seed": 0}),),
        ).lastrowid
        conn.execute(
            "INSERT INTO ejecuciones (ingesta_id, configuracion_id, seed, estado, "
            "run_id) VALUES (?, ?, 0, 'pendiente', 'encolada')",
            (ingesta, configuracion),
        )
    assert _origen(capsys, root, db, sha)[1]["datos_sinteticos"] is False
    _conflicto(_origen(capsys, root, db, sha, "--datos-sinteticos"))
    assert correr(db, root, "sintetica", "--csv", str(csv), "--datos-sinteticos") == 1
    assert _corridas(db) == []
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO configuraciones (version, parametros) VALUES (1, ?)",
            (json.dumps({"datos_sinteticos": True}),),
        )
        conn.execute(
            "INSERT INTO ejecuciones (ingesta_id, configuracion_id, seed, estado, "
            "run_id) VALUES (?, last_insert_rowid(), 0, 'pendiente', 'otra')",
            (ingesta,),
        )
    # Dos configuraciones que se contradicen bastan para rechazar la consulta.
    _conflicto(_origen(capsys, root, db, sha))


def test_origen_es_de_solo_lectura_y_rechaza_entradas_invalidas(tmp_path, capsys):
    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "pred.db")
    antes = db.read_bytes()
    codigo, cuerpo = _origen(capsys, root, db, "a" * 64, "--datos-reales")
    assert (codigo, cuerpo["datos_sinteticos"]) == (0, False)
    assert db.read_bytes() == antes
    assert not root.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pred.db"]
    capsys.readouterr()
    assert cli.main(["origen", "--ingest-id", "no-es-sha", "--db-path", str(db)]) == 1
    ausente = tmp_path / "ausente.db"
    assert cli.main(["origen", "--ingest-id", "a" * 64, "--db-path", str(ausente)]) == 1
    assert not ausente.exists()
    assert "base de la plataforma" in capsys.readouterr().err


def test_origen_como_subproceso_imprime_una_linea_json(tmp_path):
    db = crear_base_plataforma(tmp_path / "pred.db")
    inicio = time.monotonic()
    hijo = subprocess.run(
        [
            sys.executable,
            "-m",
            "pred_engine.cli",
            "origen",
            "--ingest-id",
            "b" * 64,
            "--data-root",
            str(tmp_path / "data"),
            "--db-path",
            str(db),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PRED_DATA_ROOT": str(tmp_path / "otra")},
    )
    assert time.monotonic() - inicio < 60
    assert hijo.returncode == 0, hijo.stderr
    assert json.loads(hijo.stdout) == {"ingest_id": "b" * 64, "datos_sinteticos": None}
