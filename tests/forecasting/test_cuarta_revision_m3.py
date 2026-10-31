"""Origen M1 en la biblioteca nativa y ajustes de una ejecucion central."""

import json
import sqlite3

import pytest
from tests._plataforma import crear_base_plataforma
from tests.forecasting.test_tercera_revision_m3 import fuente

from pred_engine.cli import main
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


def evaluar_parquet(root, db, datos_sinteticos):
    """La composicion documentada: M1 por CLI y evaluacion por la biblioteca."""
    source = fuente(root.parent)
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
    parquet = root / json.loads(report.read_text())["published"]["parquet_path"]
    with abrir_base_plataforma(db) as conn:
        validator = ControladorEvaluacion(
            conn,
            root / "runs/biblioteca",
            run_id_m2="biblioteca",
            codigo_version="prueba",
            datos_sinteticos=datos_sinteticos,
            data_root=root,
        )
        # La serie es smooth: ML y DL cumplen la regla de dos familias.
        pipeline = build_pipeline(
            EvaluationSettings(min_train=60, horizon=7, step=28),
            families=("ml", "dl"),
            n_trials=1,
            seed=17,
            workers=1,
            validator=validator,
        )
        return pipeline.run(
            PipelineInput(
                parquet_path=parquet,
                data_root=root,
                datos_sinteticos=datos_sinteticos,
            )
        )


def test_biblioteca_rechaza_declarar_reales_datos_sinteticos_de_m1(tmp_path):
    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "central.db")
    with pytest.raises(
        PipelineExecutionError, match="L4: el origen .* no puede cambiar"
    ):
        evaluar_parquet(root, db, False)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM ejecuciones").fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM m3_corridas").fetchone() == (0,)


def test_biblioteca_sin_declaracion_conserva_el_origen_sintetico_de_m1(tmp_path):
    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "central.db")
    result = evaluar_parquet(root, db, None)
    assert result.validation.audit_bundle["datos_sinteticos"] is True
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT estado, datos_sinteticos, origen_declarado FROM m3_corridas"
        ).fetchall() == [("PUBLICADA", 1, 1)]
        assert conn.execute(
            "SELECT COUNT(*) FROM m3_veredictos WHERE dm_p_valor IS NOT NULL "
            "OR veredicto = 'VALIDADO'"
        ).fetchone() == (0,)


def correr(source, root, db, runs, *extra):
    return main(
        [
            "run",
            "--csv",
            str(source),
            "--data-root",
            str(root),
            "--db-path",
            str(db),
            "--runs-dir",
            str(runs),
            "--run-id",
            "compartida",
            "--families",
            "ml",
            "dl",
            "--trials",
            "1",
            "--min-train",
            "60",
            "--step",
            "28",
            "--seed",
            "17",
            "--datos-sinteticos",
            "--telemetry-interval",
            "0",
            *extra,
        ]
    )


def test_ejecucion_central_rechaza_otro_horizonte_con_el_mismo_run_id(tmp_path):
    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "central.db")
    source = fuente(tmp_path)
    assert correr(source, root, db, tmp_path / "runs-a", "--workers", "1") == 0
    consultas = {
        "ejecuciones": "SELECT e.estado, e.error_json, c.parametros FROM ejecuciones e "
        "JOIN configuraciones c ON c.id = e.configuracion_id",
        "m3_corridas": "SELECT run_id, estado FROM m3_corridas",
        "m3_protocolos": "SELECT run_id, protocolo FROM m3_protocolos",
    }
    with sqlite3.connect(db) as conn:
        antes = {t: conn.execute(q).fetchall() for t, q in consultas.items()}
    # Otro directorio local no protege la ejecucion central: H=5 se rechaza.
    assert correr(source, root, db, tmp_path / "runs-b", "--horizon", "5") == 1
    with sqlite3.connect(db) as conn:
        assert {t: conn.execute(q).fetchall() for t, q in consultas.items()} == antes
    assert json.loads(antes["ejecuciones"][0][2])["horizon"] == 7
    # Solo cambia la ejecucion paralela: la misma ejecucion se reutiliza.
    assert correr(source, root, db, tmp_path / "runs-c", "--workers", "2") == 0
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM ejecuciones").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM m3_corridas").fetchone() == (1,)
