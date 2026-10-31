"""Calendario finito y reintentos de corridas publicadas de la tercera revision."""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
from tests._plataforma import crear_base_plataforma
from tests._series import serie_ar_estacional

from pred_engine import pipeline as modulo_pipeline
from pred_engine.cli import main
from pred_engine.unit_checkpoints import load_result

pytestmark = pytest.mark.slow


def fuente(tmp_path, *, infinito=None, demanda=None):
    path = tmp_path / "source.csv"
    if demanda is None:
        demanda = 30 + np.arange(220) * 0.05 + np.sin(np.arange(220))
    frame = pd.DataFrame(
        {
            "sku_id": ["SKU"] * 220,
            "timestamp": pd.date_range("2024-01-01", periods=220),
            "demand_qty": demanda,
            "lead_time_days": 4,
        }
    )
    if infinito is not None:
        frame.loc[frame.timestamp == infinito, "demand_qty"] = np.inf
    frame.to_csv(path, index=False)
    return path


def ejecutar(source, root, db, *familias):
    return main(
        [
            "run",
            "--csv",
            str(source),
            "--data-root",
            str(root),
            "--db-path",
            str(db),
            "--run-id",
            "corrida",
            "--families",
            *familias,
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
            "--datos-sinteticos",
            "--telemetry-interval",
            "0",
        ]
    )


def test_valor_no_finito_reservado_omite_contexto_sin_contar_fallos(tmp_path):
    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "central.db")
    source = fuente(tmp_path, infinito="2024-07-20")
    # La serie es smooth: ML y DL cumplen la regla de dos familias.
    assert ejecutar(source, root, db, "ml", "dl") == 0
    # Un reintento identico reutiliza la publicacion en lugar de quedar a medias.
    assert ejecutar(source, root, db, "ml", "dl") == 0
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT estado, causa_rechazo FROM m3_corridas"
        ).fetchall() == [("PUBLICADA", None)]
        previstas, procedencia = conn.execute(
            "SELECT n_unidades_previstas, procedencia FROM m3_protocolos"
        ).fetchone()
        assert conn.execute(
            "SELECT estado, COUNT(*) FROM m3_unidades GROUP BY estado"
        ).fetchall() == [("confirmada", previstas)]
        assert conn.execute(
            "SELECT estado, error_json FROM ejecuciones"
        ).fetchall() == [("completada", None)]
        horizonte = {
            n
            for (n,) in conn.execute(
                "SELECT COUNT(*) FROM m3_pronosticos GROUP BY run_id, sku, "
                "candidato_id, origen"
            )
        }
    # t* = 2024-06-24: 26 origenes con contexto finito, dos candidatos y linea base,
    # H=7.
    assert previstas == 3 * 26 and horizonte == {7}
    exclusiones = json.loads(procedencia)["exclusiones_datos"]["SKU"]
    causas = {e["origen"]: e["causa"] for e in exclusiones}
    assert [o for o, c in causas.items() if c == "sin_datos_objetivo"] == [
        f"2024-07-{d:02d}T00:00:00" for d in range(13, 20)
    ]
    assert min(o for o, c in causas.items() if c == "sin_contexto_observado") == (
        "2024-07-20T00:00:00"
    )


def test_reintento_publicado_sin_checkpoint_de_ajuste_conserva_ejecucion(
    tmp_path, monkeypatch
):
    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "central.db")
    # Una serie determinista deja la raiz MA estacional de SARIMA en el borde y
    # el ajuste se rechaza (ADR-020): la familia clasica necesita ruido.
    source = fuente(tmp_path, demanda=np.round(serie_ar_estacional(220), 2))
    assert ejecutar(source, root, db, "classical", "ml") == 0
    tablas = ("m3_corridas", "m3_unidades", "m3_pronosticos", "m3_veredictos")
    with sqlite3.connect(db) as conn:
        antes = {t: conn.execute(f"SELECT * FROM {t}").fetchall() for t in tablas}
        ejecucion = conn.execute(
            "SELECT estado, error_json FROM ejecuciones"
        ).fetchall()
    assert ejecucion == [("completada", None)]
    # Sin el checkpoint ML, esa unidad necesita ajustarse de nuevo. Las demas
    # deben restaurarse; la publicacion M3 previa sigue siendo inmutable.
    for path in (root / "runs/corrida/checkpoints/L2").glob("*.json"):
        checkpoint = load_result(path)
        if checkpoint.valor.selection.family == "ml":
            path.unlink()
    original = modulo_pipeline._fit_unit
    called = []

    def ajuste_transitorio(task):
        called.append(task.decision.family)
        if task.decision.family == "ml":
            raise RuntimeError("dependencia no disponible durante el ajuste final")
        return original(task)

    monkeypatch.setattr(modulo_pipeline, "_fit_unit", ajuste_transitorio)
    # El fallo aislado de M2 se informa; la evidencia y la ejecucion publicadas no.
    assert ejecutar(source, root, db, "classical", "ml") == 8
    assert called == ["ml"]
    with sqlite3.connect(db) as conn:
        assert {t: conn.execute(f"SELECT * FROM {t}").fetchall() for t in tablas} == (
            antes
        )
        assert (
            conn.execute("SELECT estado, error_json FROM ejecuciones").fetchall()
            == ejecucion
        )


def test_ejemplo_de_composicion_documentado_publica_m3(tmp_path, monkeypatch):
    from pathlib import Path

    readme = Path(__file__).parents[2] / "docs/features/backend-pipeline/README.md"
    bloque = readme.read_text().split("## Composition", 1)[1]
    codigo = bloque.split("```python\n", 1)[1].split("```", 1)[0]
    monkeypatch.chdir(tmp_path)
    crear_base_plataforma(tmp_path / "data/pred.db")
    # Las dos familias del ejemplo deben entregar candidato: sin ruido, SARIMA
    # no ajusta y el unico SKU sale con la regla de dos familias (ADR-023).
    demanda = np.round(serie_ar_estacional(220), 2)
    fuente(tmp_path, demanda=demanda).rename(tmp_path / "sales.csv")
    espacio: dict[str, object] = {"__name__": "__main__"}
    exec(compile(codigo, str(readme), "exec"), espacio)
    assert espacio["result"].validation.audit_bundle["estado"] == "PUBLICADA"
    with sqlite3.connect(tmp_path / "data/pred.db") as conn:
        assert conn.execute(
            "SELECT estado, datos_sinteticos, origen_declarado FROM m3_corridas"
        ).fetchall() == [("PUBLICADA", 0, 1)]


def test_lecturas_publicadas_distinguen_origen_desconocido(tmp_path):
    from tests.forecasting.test_segunda_revision_m3 import ejecutar as ejecutar_m1

    from pred_engine.forecasting.persistencia_evidencia import veredictos_publicados
    from pred_engine.run_artifacts import sha256_file

    root, db = tmp_path / "data", crear_base_plataforma(tmp_path / "central.db")
    source = fuente(tmp_path)
    assert main(["classify", "--csv", str(source), "--data-root", str(root)]) == 0
    report = root / "ingestas" / f"{sha256_file(source)}.json"
    parquet = root / json.loads(report.read_text())["published"]["parquet_path"]
    assert ejecutar_m1(parquet, root, db, "unknown", parquet=True) == 0
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT sku, datos_sinteticos, origen_declarado FROM evidencia_publicada"
        ).fetchall() == [("SKU", 0, 0)]
        (run_id,) = conn.execute("SELECT run_id FROM m3_corridas").fetchone()
        filas = veredictos_publicados(conn, run_id)
    assert [(f["datos_sinteticos"], f["origen_declarado"]) for f in filas] == [(0, 0)]
