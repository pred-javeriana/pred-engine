"""La ejecucion central conserva una sola receta aunque se pierdan archivos."""

import json
import sqlite3
import sys
from dataclasses import replace

import pandas as pd
import pytest
from tests._plataforma import crear_base_plataforma
from tests.forecasting.test_correcciones_m3 import caso
from tests.forecasting.test_tercera_revision_m3 import ejecutar, fuente

from pred_engine.forecasting.adaptador_candidatos import LoteRechazadoError
from pred_engine.forecasting.control_reanudacion import ControladorEvaluacion
from pred_engine.forecasting.persistencia_evidencia import EvidenciaInconsistenteError

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("interrumpida", [False, True])
def test_cli_sin_manifiesto_ni_checkpoints_no_duplica_publicacion(
    tmp_path, interrumpida
):
    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "central.db")
    source = fuente(tmp_path)

    def caer_tras_unidad(frame, event, arg):
        if frame.f_code.co_name == "guardar_unidades_generadas" and event == "return":
            raise KeyboardInterrupt("caida tras confirmar una unidad M3")
        return caer_tras_unidad

    previous = sys.gettrace()
    try:
        if interrumpida:
            sys.settrace(caer_tras_unidad)
            with pytest.raises(KeyboardInterrupt, match="confirmar una unidad"):
                ejecutar(source, root, db, "ml", "dl")
        else:
            assert ejecutar(source, root, db, "ml", "dl") == 0
    finally:
        sys.settrace(previous)
    manifest = root / "runs/corrida/candidatos.json"
    frozen = json.loads(manifest.read_text())
    assert {c["familia"] for c in frozen["candidatos"]} == {"ml", "dl"}
    with sqlite3.connect(db) as conn:
        tables = [
            name
            for (name,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'm3_%'"
            )
        ]
        before = {t: conn.execute(f"SELECT * FROM {t}").fetchall() for t in tables}
        assert conn.execute("SELECT estado FROM m3_corridas").fetchall() == [
            ("EN_CURSO" if interrumpida else "PUBLICADA",)
        ]
    manifest.unlink()
    checkpoints = list((root / "runs/corrida/checkpoints/L2").glob("*.json"))
    assert len(checkpoints) == 2
    for path in checkpoints:
        path.unlink()
    failed = []

    def fallo_final(frame, event, arg):
        if frame.f_code.co_name == "_fit_unit" and event == "line":
            selection = frame.f_locals.get("selection")
            if (
                selection is not None
                and selection.family == "dl"
                and "model" in frame.f_locals
            ):
                failed.append(selection.family)
                raise RuntimeError("dependencia no disponible durante el ajuste final")
        return fallo_final

    previous = sys.gettrace()
    try:
        sys.settrace(fallo_final)
        assert ejecutar(source, root, db, "ml", "dl") == 8
    finally:
        sys.settrace(previous)
    assert failed == ["dl"]
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT ejecucion_id, COUNT(*) FROM m3_protocolos GROUP BY ejecucion_id"
        ).fetchall() == [(1, 1)]
        assert conn.execute("SELECT estado FROM m3_corridas").fetchall() == [
            ("PUBLICADA",)
        ]
        if not interrumpida:
            assert {
                t: conn.execute(f"SELECT * FROM {t}").fetchall() for t in tables
            } == before
        else:
            assert (
                conn.execute("SELECT * FROM m3_protocolos").fetchall()
                == before["m3_protocolos"]
            )
            assert set(before["m3_unidades"]).issubset(
                conn.execute("SELECT * FROM m3_unidades").fetchall()
            )
        assert conn.execute(
            "SELECT estado, COUNT(*) FROM m3_unidades GROUP BY estado"
        ).fetchall() == [("confirmada", 114)]
        # La exclusion que el fallo del reintento provocaria no cambia la receta
        # durable: SKU conserva sus dos familias y su veredicto.
        assert conn.execute("SELECT sku FROM m3_veredictos").fetchall() == [("SKU",)]
        assert conn.execute("SELECT COUNT(*) FROM m3_exclusiones").fetchone() == (0,)
        assert conn.execute(
            "SELECT estado, error_json FROM ejecuciones"
        ).fetchall() == [
            ("completada_con_fallos" if interrumpida else "completada", None)
        ]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert json.loads(manifest.read_text()) == frozen


def test_ejecucion_ambigua_no_elige_un_protocolo(tmp_path):
    conn, pipeline, request = caso(tmp_path)
    evaluation = pipeline.evaluate(pipeline.fit(pipeline.ingest(request)))
    pipeline.validate(evaluation)
    controller = pipeline.validator
    pipeline.validator = ControladorEvaluacion(
        conn,
        tmp_path / "segunda",
        run_id_m2="segunda",
        codigo_version="revision",
        datos_sinteticos=True,
        data_root=tmp_path / "data",
    )
    pipeline.validate(evaluation)
    # Estado heredado del defecto: dos protocolos bajo la misma ejecucion.
    conn.execute("UPDATE m3_protocolos SET ejecucion_id = 1")
    conn.commit()
    before = conn.execute("SELECT * FROM m3_protocolos").fetchall()
    runs = conn.execute("SELECT * FROM m3_corridas").fetchall()
    manifest = tmp_path / "corrida/candidatos.json"
    manifest.unlink()
    pipeline.validator = controller
    with pytest.raises(EvidenciaInconsistenteError, match="varios protocolos"):
        pipeline.validate(evaluation)
    assert conn.execute("SELECT * FROM m3_protocolos").fetchall() == before
    assert conn.execute("SELECT * FROM m3_corridas").fetchall() == runs
    assert not manifest.exists()
    conn.close()


@pytest.mark.parametrize("deriva", ["candidatos", "politica", "particion"])
def test_deriva_activa_sin_manifiesto_rechaza_la_receta_original(
    tmp_path, monkeypatch, deriva
):
    conn, pipeline, request = caso(tmp_path)
    evaluation = pipeline.evaluate(pipeline.fit(pipeline.ingest(request)))

    def caida(*args, **kwargs):
        raise KeyboardInterrupt("caida antes de pronosticar")

    with monkeypatch.context() as patch:
        patch.setattr(pipeline.validator, "_pronosticar", caida)
        with pytest.raises(KeyboardInterrupt, match="antes de pronosticar"):
            pipeline.validate(evaluation)
    before = conn.execute("SELECT * FROM m3_protocolos").fetchall()
    assert len(before) == 1
    (tmp_path / "corrida/candidatos.json").unlink()
    if deriva == "candidatos":
        evaluation = replace(
            evaluation, fitting=replace(evaluation.fitting, candidates=())
        )
    elif deriva == "politica":
        pipeline.validator.politica = replace(pipeline.validator.politica, n_min=99)
    else:
        fitting = evaluation.fitting
        evaluation = replace(
            evaluation,
            fitting=replace(fitting, reserve=replace(fitting.reserve, fraction=0.3)),
        )
    with pytest.raises(LoteRechazadoError, match="incompatible|conjunto M2"):
        pipeline.validate(evaluation)
    rows = conn.execute("SELECT estado, causa_rechazo FROM m3_corridas").fetchall()
    assert len(rows) == 1
    state, cause = rows[0]
    assert state == "RECHAZADA" and cause.startswith("contexto_incompatible:")
    assert conn.execute("SELECT * FROM m3_protocolos").fetchall() == before
    assert conn.execute("SELECT COUNT(*) FROM m3_unidades").fetchone() == (0,)
    conn.close()


@pytest.mark.parametrize("perder_archivos", [False, True])
def test_fallo_original_resuelto_en_el_reintento_completa_la_receta(
    tmp_path, monkeypatch, perder_archivos
):
    import pred_engine.pipeline as pipeline_module

    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "central.db")
    source = fuente(tmp_path)
    frame = pd.read_csv(source)
    pd.concat(
        [frame, frame.assign(sku_id="OTRO", demand_qty=frame.demand_qty + 100)]
    ).to_csv(source, index=False)
    original = pipeline_module._fit_unit

    def fallo_dl(task):
        fitted = original(task)
        if (fitted.selection.sku_id, fitted.selection.family) == ("SKU", "dl"):
            raise RuntimeError("dependencia no disponible durante el ajuste final")
        return fitted

    def caer_tras_unidad(frame, event, arg):
        if frame.f_code.co_name == "guardar_unidades_generadas" and event == "return":
            raise KeyboardInterrupt("caida tras confirmar una unidad M3")
        return caer_tras_unidad

    previous = sys.gettrace()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(pipeline_module, "_fit_unit", fallo_dl)
            sys.settrace(caer_tras_unidad)
            with pytest.raises(KeyboardInterrupt, match="confirmar una unidad"):
                ejecutar(source, root, db, "ml", "dl")
    finally:
        sys.settrace(previous)
    manifest = root / "runs/corrida/candidatos.json"
    frozen = json.loads(manifest.read_text())
    # SKU sale al cerrar L2 con una sola familia evaluable; OTRO conserva las dos.
    assert {(c["sku"], c["familia"]) for c in frozen["candidatos"]} == {
        ("OTRO", "ml"),
        ("OTRO", "dl"),
    }
    if perder_archivos:
        manifest.unlink()
        for path in (root / "runs/corrida/checkpoints/L2").glob("*.json"):
            path.unlink()
    assert ejecutar(source, root, db, "ml", "dl") == 0
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT estado FROM m3_corridas").fetchall() == [
            ("PUBLICADA",)
        ]
        assert conn.execute("SELECT COUNT(*) FROM m3_protocolos").fetchone() == (1,)
        # El ajuste resuelto no devuelve SKU a la competencia: manda la exclusion
        # durable del intento original.
        assert conn.execute("SELECT sku FROM m3_veredictos").fetchall() == [("OTRO",)]
        assert conn.execute("SELECT sku, codigo FROM m3_exclusiones").fetchall() == [
            ("SKU", "eligibility.fewer_than_two_families")
        ]
    assert json.loads(manifest.read_text()) == frozen


def test_deriva_sobre_publicada_conserva_la_ejecucion_completada(tmp_path, monkeypatch):
    import pred_engine.run_artifacts as run_artifacts

    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "central.db")
    source = fuente(tmp_path)
    assert ejecutar(source, root, db, "ml", "dl") == 0
    with sqlite3.connect(db) as conn:
        before = conn.execute("SELECT * FROM ejecuciones").fetchall()
    original = run_artifacts.library_versions
    monkeypatch.setattr(
        run_artifacts, "library_versions", lambda: {**original(), "numpy": "0.0"}
    )
    assert ejecutar(source, root, db, "ml", "dl") == 1
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT estado FROM m3_corridas").fetchall() == [
            ("PUBLICADA",)
        ]
        assert conn.execute("SELECT * FROM ejecuciones").fetchall() == before
