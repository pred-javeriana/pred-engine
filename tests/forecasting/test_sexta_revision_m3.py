"""Huellas compartidas, protocolo efectivo, version de codigo y bytes evaluados."""

import json
import shutil
import sqlite3
from contextlib import contextmanager

import pandas as pd
import pytest
from tests._plataforma import crear_base_plataforma
from tests.forecasting.test_quinta_revision_m3 import correr, estado
from tests.forecasting.test_tercera_revision_m3 import fuente

from pred_engine import run_artifacts
from pred_engine.forecasting.control_reanudacion import ControladorEvaluacion
from pred_engine.forecasting.persistencia_evidencia import abrir_base_plataforma
from pred_engine.pipeline import (
    EvaluationSettings,
    PipelineExecutionError,
    PipelineInput,
)
from pred_engine.pipeline_setup import build_pipeline
from pred_engine.run_artifacts import sha256_file

pytestmark = pytest.mark.slow


def informe(root, source):
    return root / "ingestas" / f"{sha256_file(source)}.json"


def parquet_de(root, source):
    return (
        root
        / json.loads(informe(root, source).read_text())["published"]["parquet_path"]
    )


def test_huella_compartida_por_dos_ingestas_conserva_su_origen_sintetico(tmp_path):
    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "central.db")
    lf = fuente(tmp_path).rename(tmp_path / "a.csv")
    crlf = tmp_path / "b.csv"
    crlf.write_bytes(lf.read_bytes().replace(b"\n", b"\r\n"))
    for source, run_id in ((lf, "a"), (crlf, "b")):
        assert correr(db, root, run_id, "--csv", str(source), "--datos-sinteticos") == 0
    assert sha256_file(parquet_de(root, lf)) == sha256_file(parquet_de(root, crlf))
    # Los mismos bytes fuera de su raiz registrada no pueden declararse reales.
    otra = tmp_path / "otra"
    otra.mkdir()
    copia = otra / "copia.parquet"
    shutil.copy2(parquet_de(root, lf), copia)
    antes = estado(db)
    assert correr(db, otra, "real", "--parquet", str(copia), "--datos-reales") == 1
    assert estado(db) == antes
    assert correr(db, otra, "sin-declarar", "--parquet", str(copia)) == 0
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT datos_sinteticos, origen_declarado FROM m3_corridas "
            "WHERE run_id_m2 = 'sin-declarar'"
        ).fetchall() == [(1, 1)]


def test_ruta_sin_huella_conocida_no_hereda_un_origen_historico(tmp_path):
    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "central.db")
    source = fuente(tmp_path)
    assert correr(db, root, "real", "--csv", str(source), "--datos-reales") == 0
    # Estado sin huellas: ni m3_ingestas ni published.sha256 identifican los bytes.
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM m3_ingestas")
    report = json.loads(informe(root, source).read_text())
    report["published"]["sha256"] = None
    informe(root, source).write_text(json.dumps(report))
    parquet = parquet_de(root, source)
    panel = pd.read_parquet(parquet)
    panel["demand_qty"] = panel["demand_qty"] * 2 + 1
    panel.to_parquet(parquet, index=False)
    assert correr(db, root, "reemplazo", "--parquet", str(parquet)) == 0
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT c.datos_sinteticos, c.origen_declarado, c.ingesta_sha256 "
            "FROM m3_corridas c WHERE c.run_id_m2 = 'reemplazo'"
        ).fetchall() == [(0, 0, sha256_file(parquet))]
        assert conn.execute(
            "SELECT COUNT(*) FROM m3_veredictos v JOIN m3_corridas c USING (run_id) "
            "WHERE c.run_id_m2 = 'reemplazo' AND (v.dm_p_valor IS NOT NULL "
            "OR v.veredicto = 'VALIDADO')"
        ).fetchone() == (0,)


@contextmanager
def biblioteca(root, db, run_id, *, configuracion_m2=None):
    with abrir_base_plataforma(db) as conn:
        controller = ControladorEvaluacion(
            conn,
            root / "runs" / run_id,
            run_id_m2=run_id,
            codigo_version="prueba",
            datos_sinteticos=None,
            data_root=root,
            configuracion_m2=configuracion_m2,
        )
        yield build_pipeline(
            EvaluationSettings(min_train=60, horizon=7, step=28),
            families=("ml",),
            n_trials=1,
            seed=17,
            workers=1,
            validator=controller,
        )


def test_protocolo_registra_los_ajustes_m2_efectivos(tmp_path):
    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "central.db")
    source = fuente(tmp_path)
    request = PipelineInput(csv_path=source, data_root=root, datos_sinteticos=True)
    contradictoria = {"families": ["classical"], "trials": 10, "seed": 42}
    with (
        biblioteca(root, db, "contradictoria", configuracion_m2=contradictoria) as p,
        pytest.raises(PipelineExecutionError, match="families, seed, trials"),
    ):
        p.run(request)
    assert estado(db) == {"ejecuciones": [], "m3_corridas": []}
    with biblioteca(root, db, "efectiva") as p:
        p.run(request)
    with sqlite3.connect(db) as conn:
        (protocolo,) = conn.execute("SELECT protocolo FROM m3_protocolos").fetchone()
    m2 = json.loads(protocolo)["configuracion_m2"]
    assert (m2["families"], m2["trials"], m2["seed"], m2["horizon"]) == (
        ["ml"],
        1,
        17,
        7,
    )


def test_otra_version_de_codigo_no_reutiliza_la_ejecucion_central(
    tmp_path, monkeypatch, capsys
):
    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "central.db")
    source = fuente(tmp_path)
    entrada = ("--csv", str(source), "--datos-sinteticos")
    assert correr(db, root, "codigo", *entrada, "--runs-dir", str(tmp_path / "a")) == 0
    antes = estado(db)
    monkeypatch.setattr(run_artifacts, "code_fingerprint", lambda: "otra-version")
    capsys.readouterr()
    assert correr(db, root, "codigo", *entrada, "--runs-dir", str(tmp_path / "b")) == 1
    assert "otra configuracion de resultados: code" in capsys.readouterr().out
    assert estado(db) == antes


def test_biblioteca_registra_la_huella_de_los_bytes_evaluados(tmp_path):
    from pred_engine.cli import main

    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "central.db")
    source = fuente(tmp_path)
    assert (
        main(
            [
                "classify",
                "--csv",
                str(source),
                "--data-root",
                str(root),
                "--datos-reales",
            ]
        )
        == 0
    )
    parquet = parquet_de(root, source)
    original = parquet.read_bytes()
    panel = pd.read_parquet(parquet)
    panel["demand_qty"] = panel["demand_qty"] * 2 + 1
    panel.to_parquet(parquet, index=False)
    reemplazo = sha256_file(parquet)
    with biblioteca(root, db, "intercambio") as p:
        ingestion = p.ingest(PipelineInput(parquet_path=parquet, data_root=root))
        parquet.write_bytes(original)  # los bytes registrados vuelven antes de L4
        p.validate(p.evaluate(p.fit(ingestion)))
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT ingesta_sha256, datos_sinteticos, origen_declarado FROM m3_corridas"
        ).fetchall() == [(reemplazo, 0, 0)]
        assert conn.execute(
            "SELECT COUNT(*) FROM m3_veredictos WHERE dm_p_valor IS NOT NULL "
            "OR veredicto = 'VALIDADO'"
        ).fetchone() == (0,)
