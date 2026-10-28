"""Huella obligatoria en la biblioteca nativa y version de codigo efectiva."""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import replace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests._plataforma import crear_base_plataforma
from tests.forecasting.test_quinta_revision_m3 import correr, estado
from tests.forecasting.test_sexta_revision_m3 import parquet_de
from tests.forecasting.test_tercera_revision_m3 import fuente

from pred_engine.cli import main
from pred_engine.forecasting.control_reanudacion import ControladorEvaluacion
from pred_engine.forecasting.persistencia_evidencia import abrir_base_plataforma
from pred_engine.ingesta.salida import read_classified_publication
from pred_engine.pipeline import (
    EvaluationSettings,
    PipelineExecutionError,
    PipelineInput,
    ingestion_artifact_from_publication,
)
from pred_engine.pipeline_setup import build_pipeline

pytestmark = pytest.mark.slow


@contextmanager
def biblioteca(root, db, run_id, codigo, *, configuracion_m2=None):
    with abrir_base_plataforma(db) as conn:
        yield build_pipeline(
            EvaluationSettings(min_train=60, horizon=7, step=28),
            families=("ml",),
            n_trials=1,
            seed=17,
            workers=1,
            validator=ControladorEvaluacion(
                conn,
                root / "runs" / run_id,
                run_id_m2=run_id,
                codigo_version=codigo,
                datos_sinteticos=None,
                data_root=root,
                configuracion_m2=configuracion_m2,
            ),
        )


def test_artefacto_sin_huella_no_hereda_la_huella_del_archivo_restaurado(tmp_path):
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
    meta = pq.read_table(parquet).schema.metadata
    panel = pd.read_parquet(parquet)
    panel["demand_qty"] = panel["demand_qty"] * 2 + 1
    pq.write_table(pa.Table.from_pandas(panel).replace_schema_metadata(meta), parquet)
    # Construccion manual del artefacto: no lleva la huella de los bytes leidos.
    ingestion = replace(
        ingestion_artifact_from_publication(
            parquet, read_classified_publication(parquet)
        ),
        parquet_sha256=None,
    )
    parquet.write_bytes(original)
    with biblioteca(root, db, "manual", "prueba") as p:
        evaluation = p.evaluate(p.fit(ingestion))
        with pytest.raises(ValueError, match="parquet_sha256"):
            p.validate(evaluation)
    assert estado(db) == {"ejecuciones": [], "m3_corridas": []}


def test_biblioteca_reconcilia_la_version_de_codigo_efectiva(tmp_path):
    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "central.db")
    source = fuente(tmp_path)
    request = PipelineInput(csv_path=source, data_root=root, datos_sinteticos=True)
    with biblioteca(root, db, "biblioteca", "version-a") as p:
        p.run(request)
    with sqlite3.connect(db) as conn:
        (protocolo,) = conn.execute("SELECT protocolo FROM m3_protocolos").fetchone()
    assert json.loads(protocolo)["configuracion_m2"]["code"] == "version-a"
    # El mismo codigo desde otro directorio reutiliza la ejecucion central.
    with biblioteca(tmp_path / "otra", db, "biblioteca", "version-a") as p:
        p.run(request)
    antes = estado(db)
    assert len(antes["ejecuciones"]) == 1
    # Otra version de codigo bajo la misma ejecucion se rechaza.
    with (
        biblioteca(tmp_path / "b", db, "biblioteca", "version-b") as p,
        pytest.raises(PipelineExecutionError, match="resultados: code"),
    ):
        p.run(request)
    # Una declaracion del llamador que contradice el codigo efectivo tambien.
    with (
        biblioteca(
            root, db, "obsoleta", "actual", configuracion_m2={"code": "obsoleta"}
        ) as p,
        pytest.raises(PipelineExecutionError, match="efectiva: code"),
    ):
        p.run(request)
    assert estado(db) == antes


def test_biblioteca_no_reutiliza_una_ejecucion_del_cli_con_otro_codigo(tmp_path):
    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "central.db")
    source = fuente(tmp_path)
    assert correr(db, root, "cli", "--csv", str(source), "--datos-sinteticos") == 0
    antes = estado(db)
    request = PipelineInput(csv_path=source, data_root=root, datos_sinteticos=True)
    with (
        biblioteca(tmp_path / "b", db, "cli", "version-b") as p,
        pytest.raises(PipelineExecutionError, match="resultados: code"),
    ):
        p.run(request)
    assert estado(db) == antes
