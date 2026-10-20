"""Identidad de contenido, origen de entrada y ajustes efectivos de M2."""

import json
import shutil
import sqlite3

import pandas as pd
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

CONSULTAS = {
    "ejecuciones": "SELECT e.run_id, e.seed, e.estado, c.parametros FROM ejecuciones e "
    "JOIN configuraciones c ON c.id = e.configuracion_id",
    "m3_corridas": "SELECT run_id, estado, datos_sinteticos, origen_declarado "
    "FROM m3_corridas",
}


def estado(db):
    with sqlite3.connect(db) as conn:
        return {t: conn.execute(q).fetchall() for t, q in CONSULTAS.items()}


def correr(db, root, run_id, *entrada):
    return main(
        [
            "run",
            *entrada,
            "--data-root",
            str(root),
            "--db-path",
            str(db),
            "--run-id",
            run_id,
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
            "--seed",
            "17",
            "--telemetry-interval",
            "0",
        ]
    )


def test_parquet_registrado_con_otro_contenido_no_hereda_su_origen_real(
    tmp_path, capsys
):
    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "central.db")
    source = fuente(tmp_path)
    assert correr(db, root, "real", "--csv", str(source), "--datos-reales") == 0
    report = json.loads((root / "ingestas" / f"{sha256_file(source)}.json").read_text())
    parquet = root / report["published"]["parquet_path"]
    # El mismo archivo con otras demandas ya no es la ingesta declarada real.
    panel = pd.read_parquet(parquet)
    panel["demand_qty"] = panel["demand_qty"] * 2 + 1
    panel.to_parquet(parquet, index=False)
    antes = estado(db)
    capsys.readouterr()
    assert correr(db, root, "reemplazo", "--parquet", str(parquet)) == 1
    assert "cambio de contenido" in capsys.readouterr().err
    assert estado(db) == antes


def ejecutar_biblioteca(root, db, parquet, run_id, entrada, validador, **m2):
    with abrir_base_plataforma(db) as conn:
        controller = ControladorEvaluacion(
            conn,
            root / "runs" / run_id,
            run_id_m2=run_id,
            codigo_version="prueba",
            datos_sinteticos=validador,
            data_root=root,
        )
        pipeline = build_pipeline(
            EvaluationSettings(min_train=60, horizon=7, step=28),
            families=("ml",),
            workers=1,
            validator=controller,
            **m2,
        )
        return pipeline.run(
            PipelineInput(
                parquet_path=parquet, data_root=root, datos_sinteticos=entrada
            )
        )


def parquet_externo(tmp_path):
    """Un Parquet clasificado copiado a otra raiz, sin informe M1 ni registro."""
    origen = tmp_path / "origen"
    source = fuente(tmp_path)
    assert main(["classify", "--csv", str(source), "--data-root", str(origen)]) == 0
    report = json.loads(
        (origen / "ingestas" / f"{sha256_file(source)}.json").read_text()
    )
    root = tmp_path / "externo"
    root.mkdir()
    parquet = root / "externo.parquet"
    shutil.copy2(origen / report["published"]["parquet_path"], parquet)
    return root, parquet


def test_parquet_externo_conserva_su_declaracion_sintetica_de_entrada(tmp_path):
    root, parquet = parquet_externo(tmp_path)
    db = crear_base_plataforma(tmp_path / "central.db")
    m2 = {"n_trials": 1, "seed": 17}
    with pytest.raises(
        PipelineExecutionError, match="L4: el origen .* no puede cambiar"
    ):
        ejecutar_biblioteca(root, db, parquet, "conflicto", True, False, **m2)
    assert estado(db) == {"ejecuciones": [], "m3_corridas": []}
    ejecutar_biblioteca(root, db, parquet, "sintetico", True, None, **m2)
    assert [fila[1:] for fila in estado(db)["m3_corridas"]] == [("PUBLICADA", 1, 1)]


def test_biblioteca_registra_los_ajustes_m2_que_produjeron_los_resultados(tmp_path):
    root, parquet = parquet_externo(tmp_path)
    db = crear_base_plataforma(tmp_path / "central.db")
    ejecutar_biblioteca(root, db, parquet, "ajustes", True, None, n_trials=1, seed=17)
    antes = estado(db)
    ((run_id, semilla, _, parametros),) = antes["ejecuciones"]
    assert (run_id, semilla) == ("ajustes", 17)
    guardados = json.loads(parametros)
    assert (guardados["families"], guardados["trials"], guardados["seed"]) == (
        ["ml"],
        1,
        17,
    )
    # Otra semilla y otro presupuesto bajo el mismo run_id no reutilizan la ejecucion.
    otra = tmp_path / "otra"
    otra.mkdir()
    with pytest.raises(PipelineExecutionError, match="seed, trials"):
        ejecutar_biblioteca(
            otra, db, parquet, "ajustes", True, None, n_trials=2, seed=23
        )
    assert estado(db) == antes
