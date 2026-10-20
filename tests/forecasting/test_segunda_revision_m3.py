"""Fronteras publicas de procedencia y posesion de la segunda revision."""

import json
import sqlite3
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from tests._evidencia import INGESTA, base, handoff, identidad
from tests._plataforma import crear_base_plataforma
from tests.forecasting.test_correcciones_m3 import caso

from pred_engine.cli import main
from pred_engine.forecasting.adaptador_candidatos import LoteRechazadoError
from pred_engine.forecasting.control_reanudacion import ControladorEvaluacion
from pred_engine.forecasting.persistencia_evidencia import registrar_corrida
from pred_engine.run_artifacts import sha256_file

pytestmark = pytest.mark.slow


def fuente(tmp_path):
    path = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "sku_id": ["SKU"] * 220,
            "timestamp": pd.date_range("2024-01-01", periods=220),
            "demand_qty": 30 + np.arange(220) * 0.05 + np.sin(np.arange(220)),
            "lead_time_days": 4,
        }
    ).to_csv(path, index=False)
    return path


def ejecutar(source, root, db, rid, *flags, parquet=False):
    return main(
        [
            "run",
            "--parquet" if parquet else "--csv",
            str(source),
            "--data-root",
            str(root),
            "--db-path",
            str(db),
            "--run-id",
            rid,
            "--families",
            "ml",
            "--trials",
            "1",
            "--min-train",
            "60",
            "--step",
            "28",
            "--workers",
            "1",
            "--telemetry-interval",
            "0",
            *flags,
        ]
    )


@pytest.mark.parametrize(
    "flags, expected",
    [([], None), (["--datos-reales"], False), (["--datos-sinteticos"], True)],
)
def test_m1_solo_declara_un_origen_explicito(tmp_path, flags, expected):
    source = fuente(tmp_path)
    root = tmp_path / "data"
    assert (
        main(["classify", "--csv", str(source), "--data-root", str(root), *flags]) == 0
    )
    report = json.loads((root / "ingestas" / f"{sha256_file(source)}.json").read_text())
    assert report["source_file"]["datos_sinteticos"] is expected


def test_parquet_de_origen_desconocido_publica_sin_inferencia(tmp_path):
    source, root = fuente(tmp_path), tmp_path / "data"
    db = crear_base_plataforma(tmp_path / "central.db")
    assert main(["classify", "--csv", str(source), "--data-root", str(root)]) == 0
    path = root / "ingestas" / f"{sha256_file(source)}.json"
    report = json.loads(path.read_text())
    parquet = root / report["published"]["parquet_path"]
    (root / "ingestas/unrelated.json").write_text("{bad")
    assert ejecutar(parquet, root, db, "unknown", parquet=True) == 0
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT estado, origen_declarado FROM m3_corridas"
        ).fetchall() == [("PUBLICADA", 0)]
        assert conn.execute(
            "SELECT veredicto, dm_p_valor, dm_p_ajustado, dm_causa FROM m3_veredictos"
        ).fetchall() == [("EXPLORATORIO", None, None, "origen_desconocido")]
        protocol = json.loads(
            conn.execute("SELECT protocolo FROM m3_protocolos").fetchone()[0]
        )
        assert protocol["datos_sinteticos"] is None


def test_csv_reusa_origen_y_rechaza_conflicto_antes_de_m1_hpo(tmp_path):
    source, root = fuente(tmp_path), tmp_path / "data"
    db = crear_base_plataforma(tmp_path / "central.db")
    assert (
        main(
            [
                "classify",
                "--csv",
                str(source),
                "--data-root",
                str(root),
                "--datos-sinteticos",
            ]
        )
        == 0
    )
    report = root / "ingestas" / f"{sha256_file(source)}.json"
    before = report.read_bytes()
    assert ejecutar(source, root, db, "conflict", "--datos-reales") == 1
    assert report.read_bytes() == before
    assert not (root / "runs/conflict").exists()
    assert ejecutar(source, root, db, "synthetic") == 0
    assert json.loads(report.read_text())["source_file"]["datos_sinteticos"] is True
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT datos_sinteticos, origen_declarado FROM m3_corridas"
        ).fetchall() == [(1, 1)]
        assert conn.execute(
            "SELECT veredicto, dm_p_valor FROM m3_veredictos"
        ).fetchall() == [("EXPLORATORIO", None)]


def test_colision_de_otra_ingesta_no_finaliza_la_ejecucion_ajena(tmp_path):
    conn, pipeline, request = caso(tmp_path)
    evaluation = pipeline.evaluate(pipeline.fit(pipeline.ingest(request)))
    pipeline.validate(evaluation)
    before = conn.execute("SELECT * FROM ejecuciones").fetchall()
    evidence = conn.execute("SELECT * FROM m3_corridas").fetchall()
    second = tmp_path / "other.csv"
    frame = pd.read_csv(request.csv_path)
    frame["demand_qty"] += 1
    frame.to_csv(second, index=False)
    another = replace(request, csv_path=second, data_root=tmp_path / "other")
    different = pipeline.evaluate(pipeline.fit(pipeline.ingest(another)))
    with pytest.raises(ValueError, match="otra ingesta"):
        pipeline.validate(different)
    assert conn.execute("SELECT * FROM ejecuciones").fetchall() == before
    assert conn.execute("SELECT * FROM m3_corridas").fetchall() == evidence
    assert pipeline.validate(evaluation).audit_bundle["estado"] == "PUBLICADA"
    conn.close()


def test_rechazo_de_receta_activa_es_terminal_y_no_modifica_publicadas():
    conn = base()
    run_id = registrar_corrida(
        conn, identidad(), handoff(), ingesta_sha256=INGESTA, datos_sinteticos=True
    )
    error = LoteRechazadoError("contexto_incompatible", "el conjunto M2 cambio")
    registrar_corrida(
        conn,
        identidad(),
        error,
        ingesta_sha256=INGESTA,
        datos_sinteticos=True,
        run_id_m2=handoff().run_id_m2,
    )
    assert conn.execute(
        "SELECT estado, causa_rechazo FROM m3_corridas WHERE run_id=?", (run_id,)
    ).fetchone() == ("RECHAZADA", "contexto_incompatible: el conjunto M2 cambio")
    conn.close()


def test_receta_distinta_no_reescribe_evidencia_publicada(tmp_path):
    conn, pipeline, request = caso(tmp_path)
    evaluation = pipeline.evaluate(pipeline.fit(pipeline.ingest(request)))
    pipeline.validate(evaluation)
    before = {
        t: conn.execute(f"SELECT * FROM {t}").fetchall()
        for t in (
            "m3_corridas",
            "m3_unidades",
            "m3_pronosticos",
            "m3_veredictos",
            "m3_protocolos",
        )
    }
    different = replace(evaluation, fitting=replace(evaluation.fitting, candidates=()))
    with pytest.raises(LoteRechazadoError, match="conjunto M2"):
        pipeline.validate(different)
    assert {t: conn.execute(f"SELECT * FROM {t}").fetchall() for t in before} == before
    conn.close()


def test_informe_canonico_seleccionado_se_valida_estrictamente(tmp_path):
    source, root = fuente(tmp_path), tmp_path / "data"
    db = crear_base_plataforma(tmp_path / "central.db")
    assert main(["classify", "--csv", str(source), "--data-root", str(root)]) == 0
    path = root / "ingestas" / f"{sha256_file(source)}.json"
    parquet = root / json.loads(path.read_text())["published"]["parquet_path"]
    path.write_text("{bad")
    assert ejecutar(parquet, root, db, "bad", parquet=True) == 1
    assert not (root / "runs/bad").exists()


def test_receta_local_incompatible_antes_de_m3_queda_rechazada(tmp_path):
    from datetime import UTC, datetime

    from pred_engine.comun.modelos.manifiesto_candidatos import ContextoParticion
    from pred_engine.optimizacion.router import construir_manifiesto

    conn, pipeline, request = caso(tmp_path)
    evaluation = pipeline.evaluate(pipeline.fit(pipeline.ingest(request)))
    fitting = evaluation.fitting
    context = ContextoParticion(
        ingesta_ref_m1=sha256_file(fitting.ingestion.parquet_path),
        t_corte_reserva=fitting.reserve.t_star.date(),
        fraccion_reserva=fitting.reserve.fraction,
    )
    path = tmp_path / "corrida/candidatos.json"
    path.parent.mkdir()
    path.write_text(
        construir_manifiesto(
            [c.selection for c in fitting.candidates],
            run_id_m2="m2",
            contexto=context,
            emitido_en=datetime.now(UTC),
        )
    )
    different = replace(evaluation, fitting=replace(fitting, candidates=()))
    with pytest.raises(LoteRechazadoError, match="conjunto M2"):
        pipeline.validate(different)
    state, cause = conn.execute(
        "SELECT estado, causa_rechazo FROM m3_corridas"
    ).fetchone()
    assert state == "RECHAZADA" and "conjunto M2" in cause
    conn.close()


def test_raiz_de_datos_es_colaborador_obligatorio(tmp_path):
    with pytest.raises(TypeError, match="data_root"):
        ControladorEvaluacion(
            sqlite3.connect(":memory:"),
            tmp_path / "data/runs/id",
            run_id_m2="id",
            codigo_version="revision",
            datos_sinteticos=None,
        )
