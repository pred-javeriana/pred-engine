"""Regresiones de identidad, reanudacion y exclusiones por datos de M3."""

import json
import sqlite3
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest
from tests._plataforma import crear_base_plataforma

from pred_engine.cli import main
from pred_engine.forecasting.adaptador_candidatos import LoteRechazadoError
from pred_engine.forecasting.control_reanudacion import ControladorEvaluacion
from pred_engine.forecasting.persistencia_evidencia import aplicar_migraciones_m3
from pred_engine.pipeline import (
    EvaluationSettings,
    PipelineInput,
    SkuVerdict,
    ValidationArtifact,
)
from pred_engine.run_artifacts import fijar_identidad
from pred_engine.verify import representative_csv, verification_pipeline

pytestmark = pytest.mark.slow


def caso(tmp_path):
    conn = sqlite3.connect(crear_base_plataforma(tmp_path / "central.db"))
    aplicar_migraciones_m3(conn)
    pipeline = verification_pipeline()
    pipeline.validator = ControladorEvaluacion(
        conn,
        tmp_path / "corrida",
        run_id_m2="m2",
        codigo_version="revision",
        datos_sinteticos=True,
        data_root=tmp_path / "data",
    )
    request = PipelineInput(
        csv_path=representative_csv(tmp_path / "source.csv"),
        data_root=tmp_path / "data",
    )
    return conn, pipeline, request


def test_validador_legacy_cubre_solo_los_skus_evaluados(tmp_path):
    p = verification_pipeline()
    csv = representative_csv(tmp_path / "source.csv")
    frame = pd.read_csv(csv)
    pd.concat(
        [frame, frame.assign(sku_id="second", demand_qty=frame.demand_qty + 100)]
    ).to_csv(csv, index=False)
    evaluation = p.evaluate(
        p.fit(p.ingest(PipelineInput(csv_path=csv, data_root=tmp_path / "data")))
    )
    partial = replace(evaluation, candidates=evaluation.candidates[:1])
    p.validator = lambda actual: ValidationArtifact(
        tuple(
            SkuVerdict(c.fitted.selection.sku_id, "partial") for c in actual.candidates
        ),
        {"literal": "evidencia de biblioteca"},
    )
    assert [(v.sku_id, v.verdict) for v in p.validate(partial).verdicts] == [
        ("proof-sku", "partial")
    ]


def test_l4_rechazada_conserva_error_y_no_declara_completada_la_ejecucion(tmp_path):
    conn, p, request = caso(tmp_path)
    evaluation = p.evaluate(p.fit(p.ingest(request)))
    path = tmp_path / "corrida/candidatos.json"
    path.parent.mkdir()
    path.write_text('{"schema_version":99}')
    with pytest.raises(LoteRechazadoError):
        p.validate(evaluation)
    estado, error = conn.execute(
        "SELECT estado, error_json FROM ejecuciones"
    ).fetchone()
    assert estado == "detenida"
    assert json.loads(error)["stage"] == "L4"
    assert json.loads(error)["code"] == "pipeline.stage_failed"
    assert (
        "schema" in json.loads(error)["detail"].lower()
        or "version" in json.loads(error)["detail"].lower()
    )
    assert conn.execute("SELECT estado FROM m3_corridas").fetchone()[0] == "RECHAZADA"
    conn.close()


def test_manifesto_durable_es_autoridad_si_cae_la_restauracion(tmp_path, monkeypatch):
    from pred_engine.forecasting.control_reanudacion import controlador

    conn, p, request = caso(tmp_path)
    evaluation = p.evaluate(p.fit(p.ingest(request)))
    first = p.validate(evaluation)
    path = tmp_path / "corrida/candidatos.json"
    original = json.loads(path.read_text())
    path.unlink()
    escribir = controlador.escribir_atomico

    def caer(destino, contenido):
        escribir(destino, contenido)
        raise KeyboardInterrupt("perdida despues de escribir")

    monkeypatch.setattr(controlador, "escribir_atomico", caer)
    with pytest.raises(KeyboardInterrupt):
        p.validate(evaluation)
    assert json.loads(path.read_text()) == original
    monkeypatch.setattr(controlador, "escribir_atomico", escribir)
    assert (
        p.validate(evaluation).audit_bundle["run_id_m3"]
        == first.audit_bundle["run_id_m3"]
    )
    assert conn.execute("SELECT estado, error_json FROM ejecuciones").fetchone() == (
        "completada",
        None,
    )
    conn.close()


def test_parquet_relativo_reusa_csv_y_ejecucion_importados(tmp_path):
    conn, p, request = caso(tmp_path)
    result = p.run(request)
    parquet = result.ingestion.parquet_path
    conn.execute(
        "UPDATE ingestas SET ruta_parquet=?",
        (str(parquet.relative_to(request.data_root)),),
    )
    conn.execute("DELETE FROM m3_ingestas")
    # La ejecucion de la plataforma declara la semilla que usa el pipeline.
    conn.execute(
        "INSERT INTO ejecuciones (ingesta_id, configuracion_id, seed, estado, run_id) "
        "VALUES (1,1,17,'pendiente','worker')"
    )
    conn.commit()
    p.validator = ControladorEvaluacion(
        conn,
        tmp_path / "worker",
        run_id_m2="worker",
        codigo_version="revision",
        datos_sinteticos=True,
        data_root=request.data_root,
    )
    replay = p.run(PipelineInput(parquet_path=parquet, data_root=request.data_root))
    assert replay.complete is True
    assert conn.execute("SELECT COUNT(*) FROM ingestas").fetchone()[0] == 1
    assert conn.execute(
        "SELECT ingesta_id, estado FROM ejecuciones WHERE run_id='worker'"
    ).fetchone() == (1, "completada")
    conn.close()


# Unidades previstas: origenes de cada SKU por sus dos candidatos y la linea base.
@pytest.mark.parametrize("cola, previstas", [(3, 108), (10, 99)])
def test_cola_sin_observaciones_no_es_fallo_tecnico(tmp_path, cola, previstas):
    conn, p, request = caso(tmp_path)
    frame = pd.DataFrame(
        {
            "sku_id": "complete",
            "timestamp": pd.date_range("2024-01-01", periods=120),
            "demand_qty": [20.0 + i % 7 for i in range(120)],
            "lead_time_days": 4,
        }
    )
    pd.concat([frame, frame.iloc[:-cola].assign(sku_id="short")]).to_csv(
        request.csv_path, index=False
    )
    p.settings = EvaluationSettings(min_train=16, horizon=7, step=7)
    result = p.run(request)
    assert result.complete is True
    assert result.validation.audit_bundle["n_fallos"] == 0
    assert result.validation.audit_bundle["n_unidades"] == previstas
    assert [
        (v.sku_id, v.evidence["comparacion_incompleta"])
        for v in result.validation.verdicts
    ] == [("complete", 0), ("short", 0)]
    assert result.validation.audit_bundle["exclusiones_datos"]["short"]
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM m3_unidades WHERE estado='fallida'"
        ).fetchone()[0]
        == 0
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM m3_pronosticos WHERE valor IS NULL"
        ).fetchone()[0]
        == 0
    )
    conn.close()


def test_m1_publica_procedencia_y_huella_para_el_consumidor_parquet(tmp_path):
    from pred_engine.ingesta.informe import ingest_csv_with_report

    csv = representative_csv(tmp_path / "source.csv")
    output = ingest_csv_with_report(
        csv, data_root=tmp_path / "data", datos_sinteticos=True
    )
    assert output.report.status == "accepted"
    assert output.report.source_file.datos_sinteticos is True
    assert output.result is not None
    assert (
        output.report.published.sha256
        == sha256(output.result.parquet_path.read_bytes()).hexdigest()
    )
    saved = json.loads(output.path.read_text())
    assert saved["source_file"]["datos_sinteticos"] is True
    assert saved["published"]["sha256"] == output.report.published.sha256


def test_identidad_sobrevive_resumen_perdido_e_intento_rechazado(tmp_path):
    settings = {"seed": 7}
    fijar_identidad(tmp_path, "original", settings)
    assert json.loads((tmp_path / "identidad.json").read_text()) == {
        "source_sha256": "original",
        "settings": {"seed": 7},
    }
    frozen = (tmp_path / "identidad.json").read_bytes()
    with pytest.raises(ValueError, match="incompatible"):
        fijar_identidad(tmp_path, "wrong", settings)
    (tmp_path / "corrida.json").write_text(
        json.dumps({"inputs": {"source": {"sha256": "wrong"}}})
    )
    fijar_identidad(tmp_path, "original", settings)
    assert (tmp_path / "identidad.json").read_bytes() == frozen


def test_no_se_adivina_identidad_de_checkpoints_sin_resumen_confiable(tmp_path):
    checkpoint = tmp_path / "hpo/sku/manifiesto.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}")
    summary = {
        "summary": {"complete": False, "units": {"L2": {"completed": 1}}},
        "inputs": {"source": {"sha256": "wrong"}},
        "settings": {"seed": 7},
    }
    (tmp_path / "corrida.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="identidad durable"):
        fijar_identidad(tmp_path, "wrong", {"seed": 7})
    assert not (tmp_path / "identidad.json").exists()
    summary["summary"]["complete"] = True
    summary["inputs"]["source"]["sha256"] = "original"
    (tmp_path / "corrida.json").write_text(json.dumps(summary))
    fijar_identidad(tmp_path, "original", {"seed": 7})
    assert (
        json.loads((tmp_path / "identidad.json").read_text())["source_sha256"]
        == "original"
    )


def test_politica_invalida_no_migra_ni_genera_m0(tmp_path, capsys):
    db = crear_base_plataforma(tmp_path / "central.db")
    seed = Path(__file__).parents[1] / "fixtures/semilla_inventario.csv"
    assert (
        main(
            [
                "run",
                "--seed-csv",
                str(seed),
                "--data-root",
                str(tmp_path / "data"),
                "--db-path",
                str(db),
                "--dm-min-origins",
                "1",
                "--no-foundation",
            ]
        )
        == 1
    )
    assert "n_min_dm" in capsys.readouterr().err
    assert not (tmp_path / "data").exists()
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 4
