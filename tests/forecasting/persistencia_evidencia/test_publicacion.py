"""Publicacion 3.5-A1: Parquet derivado de SQLite y vista para M4 (ADR-03-009/010)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from tests._evaluaciones import pronostico, serie
from tests._evidencia import base, corrida, corrida_evaluada, fallo, filas

from pred_engine.forecasting.evaluaciones.diebold_mariano import PruebaDM
from pred_engine.forecasting.persistencia_evidencia import (
    TABLAS,
    EstadoCorridaError,
    EvidenciaInconsistenteError,
    consultar,
    estado_corrida,
    guardar_evaluacion,
    guardar_unidades,
    publicar,
)


def _seleccionada(conn, *fallos, prueba_s1: PruebaDM | None = None):
    run_id, evaluaciones, resultado = corrida_evaluada(conn, *fallos)
    if prueba_s1 is not None:
        skus = tuple(
            replace(v, diebold_mariano=prueba_s1) if v.sku == "S1" else v
            for v in resultado.skus
        )
        resultado = replace(resultado, skus=skus)
    guardar_evaluacion(conn, run_id, evaluaciones, resultado)
    return run_id


def test_publicar_exige_la_corrida_seleccionada(tmp_path: Path) -> None:
    conn = base()
    run_id = corrida(conn)
    with pytest.raises(EstadoCorridaError, match="EN_CURSO"):
        publicar(conn, run_id, tmp_path)
    assert not any(tmp_path.iterdir())


def test_publicar_rechaza_si_un_sku_del_manifiesto_no_tiene_veredicto(
    tmp_path: Path,
) -> None:
    conn = base()
    run_id = _seleccionada(conn, fallo("S9"))
    with pytest.raises(EvidenciaInconsistenteError, match="S9"):
        publicar(conn, run_id, tmp_path)
    assert estado_corrida(conn, run_id) == "SELECCIONADA"
    assert not any(tmp_path.iterdir())


def test_publicar_deriva_el_parquet_de_sqlite_y_abre_la_vista(tmp_path: Path) -> None:
    conn = base()
    run_id = _seleccionada(conn, fallo())
    assert filas(conn, "evidencia_publicada") == 0

    rutas = publicar(conn, run_id, tmp_path)

    assert estado_corrida(conn, run_id) == "PUBLICADA"
    assert set(rutas) == set(TABLAS)
    evidencia = consultar(conn, run_id)
    for tabla, ruta in rutas.items():
        assert ruta.parent == tmp_path / run_id
        pd.testing.assert_frame_equal(pd.read_parquet(ruta), evidencia[tabla])
    assert not list((tmp_path / run_id).glob("*.tmp"))
    # El Parquet de la corrida ya dice PUBLICADA y trae el origen de los datos.
    corrida_parquet = pd.read_parquet(rutas["m3_corridas"])
    assert corrida_parquet["estado"].tolist() == ["PUBLICADA"]
    # Una columna toda NULL conserva el tipo declarado en el esquema.
    veredictos = pd.read_parquet(rutas["m3_veredictos"])
    assert veredictos["dm_p_valor"].isna().all()
    assert veredictos["dm_p_valor"].dtype == "Float64"
    # M4 solo ve corridas publicadas, con el origen de los datos.
    vista = pd.read_sql_query(
        "SELECT sku, datos_sinteticos FROM evidencia_publicada", conn
    )
    assert vista["sku"].tolist() == ["S1", "S2"]
    assert vista["datos_sinteticos"].tolist() == [0, 0]


def test_m4_lee_diebold_mariano_en_la_vista_publicada(tmp_path: Path) -> None:
    conn = base()
    prueba = PruebaDM(
        35,
        1,
        estadistico=-3.1,
        p_valor=0.004,
        p_ajustado=0.008,
        significativa=True,
        alfa=0.05,
    )
    run_id = _seleccionada(conn, prueba_s1=prueba)
    publicar(conn, run_id, tmp_path)
    vista = pd.read_sql_query(
        "SELECT sku, dm_n_ventanas, dm_horizonte, dm_estadistico, dm_p_valor,"
        " dm_p_ajustado, dm_alfa, dm_significativa, dm_causa"
        " FROM evidencia_publicada ORDER BY sku",
        conn,
    ).set_index("sku")
    s1 = vista.loc["S1"]
    assert s1.iloc[:7].tolist() == [35, 1, -3.1, 0.004, 0.008, 0.05, 1]
    assert pd.isna(s1["dm_causa"])
    assert vista.loc["S2", "dm_causa"] == "veredicto:EVIDENCIA_INSUFICIENTE"
    assert pd.isna(vista.loc["S2", "dm_p_ajustado"])


def test_si_la_exportacion_falla_la_corrida_no_queda_publicada(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = base()
    run_id = _seleccionada(conn)

    def _falla(*_: object) -> None:
        raise OSError("disco lleno")

    monkeypatch.setattr(
        "pred_engine.forecasting.persistencia_evidencia.repositorio._escribir_parquet",
        _falla,
    )
    with pytest.raises(OSError, match="disco lleno"):
        publicar(conn, run_id, tmp_path)
    assert estado_corrida(conn, run_id) == "SELECCIONADA"
    assert filas(conn, "evidencia_publicada") == 0


def test_una_corrida_publicada_no_se_reescribe(tmp_path: Path) -> None:
    conn = base()
    run_id = _seleccionada(conn)
    rutas = publicar(conn, run_id, tmp_path)
    marcas = {tabla: ruta.stat().st_mtime_ns for tabla, ruta in rutas.items()}

    assert publicar(conn, run_id, tmp_path) == rutas
    assert {tabla: ruta.stat().st_mtime_ns for tabla, ruta in rutas.items()} == marcas
    with pytest.raises(EstadoCorridaError, match="PUBLICADA"):
        guardar_unidades(conn, run_id, "S1", serie("S1/x", [pronostico(0, [1.0])]))
