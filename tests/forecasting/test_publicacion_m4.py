"""M3 publica con la corrida los datos que M4 muestra (migracion 10).

Se prueba en la entrada publica `pred-engine run`: corte, lanzamientos
tardios de M1, demanda observada de la reserva y la vista de campeones.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
from tests._plataforma import crear_base_plataforma
from tests.test_cli import _args_run_base

from pred_engine import cli

pytestmark = pytest.mark.slow

_T_STAR = "2024-03-20"
_RESERVA = [d.date().isoformat() for d in pd.date_range("2024-03-21", "2024-04-09")]
_TABLAS_M4 = ("m3_cortes", "m3_lanzamientos_tardios", "m3_observaciones")


def _panel_con_producto_tardio(path: Path) -> pd.DataFrame:
    """100 dias; SKU-TARDIO solo vende despues de t* = 2024-03-20 (20 reservados)."""
    fechas = pd.date_range("2024-01-01", periods=100)
    filas = [
        {
            "sku_id": sku,
            "timestamp": fecha.date().isoformat(),
            "demand_qty": demanda,
            "lead_time_days": 3,
        }
        for i, fecha in enumerate(fechas)
        for sku, demanda in (
            ("SKU-A", 20.0 + 0.5 * (i % 20)),
            ("SKU-B", 7.0 + (i % 3)),
            ("SKU-TARDIO", 0.0 if i < 80 else 15.0),
        )
    ]
    marco = pd.DataFrame(filas)
    marco.to_csv(path, index=False)
    return marco


def _consultar(db: Path, sql: str, *parametros: object) -> list[tuple]:
    with sqlite3.connect(db) as conn:
        return conn.execute(sql, parametros).fetchall()


def _conteos(db: Path) -> dict[str, int]:
    return {t: _consultar(db, f"SELECT COUNT(*) FROM {t}")[0][0] for t in _TABLAS_M4}


def _campeones_esperados(db: Path, run_id: str) -> list[tuple]:
    """El conteo de M4 recalculado en Python desde los veredictos publicados."""
    evaluados = {"VALIDADO", "EXPLORATORIO"}
    grupos: Counter = Counter()
    for clase, familia, veredicto in _consultar(
        db,
        "SELECT sku_class, familia_campeona, veredicto FROM m3_veredictos "
        "WHERE run_id = ?",
        run_id,
    ):
        for llave in ((clase, familia), (None, familia)):
            grupos[(*llave, "n")] += 1
            grupos[(*llave, veredicto in evaluados)] += 1
    return sorted(
        (
            (clase, familia, grupos[(clase, familia, "n")])
            + (grupos[(clase, familia, True)], grupos[(clase, familia, False)])
            for clase, familia, marca in grupos
            if marca == "n"
        ),
        key=lambda f: (f[0] is None, f[0] or "", f[1]),
    )


def test_run_csv_publica_corte_tardios_observaciones_y_campeones(tmp_path: Path):
    csv = tmp_path / "panel.csv"
    panel = _panel_con_producto_tardio(csv)
    raiz = tmp_path / "data"
    db = crear_base_plataforma(tmp_path / "pred.db")
    args = ["run", "--no-foundation", "--csv", str(csv), "--db-path", str(db)]
    args += ["--data-root", str(raiz)]

    assert cli.main(args) == 0

    ((run_id, estado),) = _consultar(db, "SELECT run_id, estado FROM m3_corridas")
    assert estado == "PUBLICADA"
    assert _consultar(db, "SELECT run_id, t_star FROM m3_cortes") == [(run_id, _T_STAR)]
    assert _consultar(db, "SELECT sku, causa FROM m3_lanzamientos_tardios") == [
        ("SKU-TARDIO", "lanzado despues del corte")
    ]
    skus = [s for (s,) in _consultar(db, "SELECT sku FROM m3_skus ORDER BY sku")]
    assert skus == ["SKU-A", "SKU-B"]

    # La reserva real, dia por dia y con los valores del CSV, para cada SKU evaluado.
    observadas = _consultar(
        db, "SELECT sku, fecha, valor FROM m3_observaciones ORDER BY sku, fecha"
    )
    reales = panel.set_index(["sku_id", "timestamp"])["demand_qty"]
    assert observadas == [
        (sku, fecha, reales[(sku, fecha)]) for sku in skus for fecha in _RESERVA
    ]
    assert (
        _consultar(
            db,
            "SELECT sku, fecha FROM m3_pronosticos EXCEPT "
            "SELECT sku, fecha FROM m3_observaciones",
        )
        == []
    )

    campeones = _consultar(
        db,
        "SELECT sku_class, familia, n_skus, n_evaluados, n_no_evaluados "
        "FROM m3_campeones WHERE run_id = ? "
        "ORDER BY sku_class IS NULL, sku_class, familia",
        run_id,
    )
    assert campeones == _campeones_esperados(db, run_id)
    assert sum(f[2] for f in campeones if f[0] is None) == len(skus)

    # Los derivados de la publicacion traen las tres tablas nuevas.
    corrida = json.loads(next((raiz / "runs").glob("*/corrida.json")).read_text())
    archivos = corrida["summary"]["m3"]["archivos"]
    for tabla in _TABLAS_M4:
        assert pd.read_parquet(archivos[tabla]).shape[0] == _conteos(db)[tabla]
        assert Path(archivos[tabla]).with_suffix(".csv").is_file()

    # Repetir la corrida publicada no reescribe ni duplica nada.
    antes = _conteos(db)
    assert cli.main(args) == 0
    assert _conteos(db) == antes
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_run_sobre_la_semilla_de_ci_publica_los_datos_de_m4(tmp_path: Path):
    db = crear_base_plataforma(tmp_path / "pred.db")
    raiz = tmp_path / "data"

    assert cli.main(_args_run_base(raiz, db, "ci-m4")) in (0, 8)

    ((run_id,),) = _consultar(
        db, "SELECT run_id FROM m3_corridas WHERE estado = 'PUBLICADA'"
    )
    ((t_star,),) = _consultar(db, "SELECT t_star FROM m3_cortes")
    corrida = json.loads((raiz / "runs" / "ci-m4" / "corrida.json").read_text())
    assert t_star == corrida["summary"]["reserve"]["t_star"]
    reservados = corrida["summary"]["reserve"]["reserved_days"]
    (n_skus,) = _consultar(db, "SELECT COUNT(*) FROM m3_skus")[0]
    assert _conteos(db)["m3_observaciones"] == n_skus * reservados
    assert (
        _consultar(
            db,
            "SELECT sku, fecha FROM m3_pronosticos EXCEPT "
            "SELECT sku, fecha FROM m3_observaciones",
        )
        == []
    )
    assert _consultar(
        db,
        "SELECT sku_class, familia, n_skus, n_evaluados, n_no_evaluados "
        "FROM m3_campeones ORDER BY sku_class IS NULL, sku_class, familia",
    ) == _campeones_esperados(db, run_id)
