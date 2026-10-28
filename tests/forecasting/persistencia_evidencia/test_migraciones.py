"""La actualizacion M3 conserva la base unica y nunca la crea por su cuenta."""

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from tests._plataforma import crear_base_plataforma

from pred_engine.forecasting.persistencia_evidencia import (
    BasePlataformaError,
    abrir_base_plataforma,
    aplicar_migraciones_m3,
    esquema_m3,
)
from pred_engine.forecasting.persistencia_evidencia.migraciones import (
    textos_migraciones_m3,
)


def test_migracion_desde_plataforma_y_repeticion_sin_duplicados(tmp_path: Path):
    ruta = crear_base_plataforma(tmp_path / "central.db")
    with abrir_base_plataforma(ruta) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 10
        assert aplicar_migraciones_m3(conn) == ()
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert {r[1] for r in conn.execute("PRAGMA table_info(m3_unidades)")} >= {
            "estado",
            "causa",
            "emitido_en",
            "codigo_version",
            "semilla",
        }
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = 'm3_protocolos'"
            ).fetchone()[0]
            == 1
        )


def test_una_version_atras_se_actualiza_una_vez_sin_perder_ingestas(tmp_path: Path):
    ruta = crear_base_plataforma(tmp_path / "central.db")
    conn = sqlite3.connect(ruta)
    conn.executescript(esquema_m3() + "PRAGMA user_version = 5;")
    conn.execute(
        "INSERT INTO ingestas (sha256, nombre_archivo, filas, skus) "
        "VALUES ('origen', 'origen.csv', 17, 2)"
    )
    conn.commit()
    assert aplicar_migraciones_m3(conn) == (6, 7, 8, 9, 10)
    assert conn.execute(
        "SELECT filas, skus FROM ingestas WHERE sha256 = 'origen'"
    ).fetchone() == (17, 2)
    assert aplicar_migraciones_m3(conn) == ()
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'm3_cartera'"
        ).fetchone()[0]
        == 1
    )
    conn.close()


def test_base_ausente_no_crea_archivos_ni_directorios(tmp_path: Path):
    ruta = tmp_path / "ausente" / "central.db"
    with pytest.raises(BasePlataformaError, match="base de la plataforma"):
        with abrir_base_plataforma(ruta):
            pytest.fail("no puede abrir una base ausente")
    assert not ruta.parent.exists()
    presente = crear_base_plataforma(tmp_path / "presente.db")
    with abrir_base_plataforma(presente) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 10


def test_m3_acepta_una_base_de_plataforma_en_version_11(tmp_path: Path):
    # La plataforma suma sus migraciones 0010 (publicacion para M4) y 0011 (estado
    # finalizo_m2) despues de las de M3.
    ruta = crear_base_plataforma(tmp_path / "central.db")
    conn = sqlite3.connect(ruta)
    aplicar_migraciones_m3(conn)
    conn.execute("PRAGMA user_version = 11")
    assert aplicar_migraciones_m3(conn) == ()
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 11
    conn.close()


@pytest.mark.parametrize("version", [0, 3, 12])
def test_version_incompatible_no_modifica_el_esquema(tmp_path: Path, version: int):
    ruta = crear_base_plataforma(tmp_path / "central.db")
    conn = sqlite3.connect(ruta)
    conn.execute(f"PRAGMA user_version = {version}")
    with pytest.raises(BasePlataformaError, match="version"):
        aplicar_migraciones_m3(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == version
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name GLOB 'm3_*'"
        ).fetchone()[0]
        == 0
    )
    conn.close()


def test_migracion_fallida_revierte_ddl_y_version(tmp_path: Path):
    ruta = crear_base_plataforma(tmp_path / "central.db")
    conn = sqlite3.connect(ruta)
    conn.executescript(
        esquema_m3()
        + "PRAGMA user_version = 5; CREATE TABLE m3_protocolos (otra TEXT);"
    )
    with pytest.raises(sqlite3.OperationalError, match="already exists"):
        aplicar_migraciones_m3(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 5
    assert [r[1] for r in conn.execute("PRAGMA table_info(m3_protocolos)")] == ["otra"]
    assert "estado" not in {
        r[1] for r in conn.execute("PRAGMA table_info(m3_unidades)")
    }
    conn.close()


def test_colision_despues_de_alter_revierte_todas_las_columnas(tmp_path):
    ruta = crear_base_plataforma(tmp_path / "late.db")
    with sqlite3.connect(ruta) as conn:
        conn.executescript(esquema_m3() + "PRAGMA user_version=5;")
        conn.execute(
            "CREATE TRIGGER m3_unidad_fallida_con_causa BEFORE INSERT "
            "ON m3_unidades BEGIN SELECT 1; END"
        )
        with pytest.raises(sqlite3.OperationalError, match="already exists"):
            aplicar_migraciones_m3(conn)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 5
        assert [r[1] for r in conn.execute("PRAGMA table_info(m3_unidades)")] == [
            "run_id",
            "sku",
            "candidato_id",
            "origen",
            "familia",
            "modelo",
        ]
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name='m3_protocolos'"
            ).fetchone()[0]
            == 0
        )
        conn.execute("DROP TRIGGER m3_unidad_fallida_con_causa")
        conn.commit()
        assert aplicar_migraciones_m3(conn) == (6, 7, 8, 9, 10)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 10


def test_caida_real_despues_de_alter_no_deja_ddl_parcial(tmp_path):
    ruta = crear_base_plataforma(tmp_path / "crash.db")
    with sqlite3.connect(ruta) as conn:
        conn.executescript(esquema_m3() + "PRAGMA user_version=5;")
    programa = """
import os, sqlite3, sys
from pred_engine.forecasting.persistencia_evidencia import aplicar_migraciones_m3
conn = sqlite3.connect(sys.argv[1])
def caer(sql):
    if "ADD COLUMN causa" in sql:
        os._exit(88)
conn.set_trace_callback(caer)
aplicar_migraciones_m3(conn)
"""
    child = subprocess.run([sys.executable, "-c", programa, str(ruta)], timeout=30)
    assert child.returncode == 88
    with sqlite3.connect(ruta) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 5
        assert [r[1] for r in conn.execute("PRAGMA table_info(m3_unidades)")] == [
            "run_id",
            "sku",
            "candidato_id",
            "origen",
            "familia",
            "modelo",
        ]
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name='m3_protocolos'"
            ).fetchone()[0]
            == 0
        )
        assert aplicar_migraciones_m3(conn) == (6, 7, 8, 9, 10)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 10


def test_transaccion_ajena_no_se_confirma(tmp_path: Path):
    ruta = crear_base_plataforma(tmp_path / "central.db")
    conn = sqlite3.connect(ruta)
    conn.execute(
        "INSERT INTO ingestas (sha256, nombre_archivo, filas, skus) "
        "VALUES ('pendiente', 'p.csv', 1, 1)"
    )
    with pytest.raises(BasePlataformaError, match="transaccion abierta"):
        aplicar_migraciones_m3(conn)
    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM ingestas").fetchone()[0] == 0
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 4
    conn.close()


def _base_version_9(ruta: Path) -> sqlite3.Connection:
    """Base de plataforma con las migraciones M3 hasta la 9, como antes de M4."""
    crear_base_plataforma(ruta)
    conn = sqlite3.connect(ruta)
    for version, texto in textos_migraciones_m3():
        if version <= 9:
            conn.executescript(texto)
    conn.execute("PRAGMA user_version = 9")
    conn.execute(
        "INSERT INTO ingestas (sha256, nombre_archivo, filas, skus) "
        "VALUES ('origen', 'origen.csv', 30, 4)"
    )
    return conn


def _publicada(conn: sqlite3.Connection, run_id: str, veredictos) -> None:
    conn.execute(
        "INSERT INTO m3_corridas (run_id, ingesta_sha256, identidad, "
        "datos_sinteticos, estado) VALUES (?, 'origen', '{}', 0, 'PUBLICADA')",
        (run_id,),
    )
    for sku_class in sorted({c for _, c, _, _ in veredictos}):
        conn.execute(
            "INSERT INTO m3_categorias (run_id, sku_class, familia_campeona, motivo, "
            "adverso, medianas_r, skus_comparables, excluidos, familias_excluidas, "
            "conteos, porcentajes, n_adversos) "
            "VALUES (?, ?, 'classical', 'm', 0, '{}', '[]', '{}', '{}', '{}', '{}', 0)",
            (run_id, sku_class),
        )
    for sku, sku_class, veredicto, familia in veredictos:
        conn.execute("INSERT INTO m3_skus (run_id, sku) VALUES (?, ?)", (run_id, sku))
        conn.execute(
            "INSERT INTO m3_veredictos (run_id, sku, sku_class, veredicto, "
            "familia_campeona, n_ventanas, cobertura, comparacion_incompleta, "
            "dm_n_ventanas, dm_horizonte, dm_causa, justificacion, modelos_evaluados) "
            "VALUES (?, ?, ?, ?, ?, 30, 1.0, 0, 30, 7, 'pocas_ventanas', 'j', '[]')",
            (run_id, sku, sku_class, veredicto, familia),
        )


def test_version_9_con_corridas_publicadas_pasa_a_10_y_cuenta_campeones(tmp_path):
    conn = _base_version_9(tmp_path / "central.db")
    _publicada(
        conn,
        "m3-a",
        [
            ("S1", "smooth", "VALIDADO", "classical"),
            ("S2", "smooth", "FALLO_TECNICO", "classical"),
            ("S3", "smooth", "EXPLORATORIO", "ml"),
            ("S4", "lumpy", "NO_EVALUABLE", "classical"),
        ],
    )
    _publicada(conn, "m3-b", [("S1", "erratic", "EVIDENCIA_INSUFICIENTE", "dl")])
    conn.commit()

    assert aplicar_migraciones_m3(conn) == (10,)

    filas = conn.execute(
        "SELECT run_id, sku_class, familia, n_skus, n_evaluados, n_no_evaluados "
        "FROM m3_campeones ORDER BY run_id, sku_class IS NULL, sku_class, familia"
    ).fetchall()
    assert filas == [
        ("m3-a", "lumpy", "classical", 1, 0, 1),
        ("m3-a", "smooth", "classical", 2, 1, 1),
        ("m3-a", "smooth", "ml", 1, 1, 0),
        ("m3-a", None, "classical", 3, 1, 2),
        ("m3-a", None, "ml", 1, 1, 0),
        ("m3-b", "erratic", "dl", 1, 0, 1),
        ("m3-b", None, "dl", 1, 0, 1),
    ]
    # Una corrida publicada antes de M4 no tiene corte: M4 responde metadata_missing.
    assert conn.execute("SELECT COUNT(*) FROM m3_cortes").fetchone()[0] == 0
    conn.close()


def test_lanzamiento_tardio_y_sku_evaluado_se_excluyen_mutuamente(tmp_path):
    conn = _base_version_9(tmp_path / "central.db")
    _publicada(conn, "m3-a", [("S1", "smooth", "VALIDADO", "classical")])
    conn.commit()
    aplicar_migraciones_m3(conn)
    conn.execute("INSERT INTO m3_cortes (run_id, t_star) VALUES ('m3-a', '2024-01-14')")
    conn.execute(
        "INSERT INTO m3_lanzamientos_tardios (run_id, sku, causa) "
        "VALUES ('m3-a', 'TARDE', 'lanzado despues del corte')"
    )
    with pytest.raises(sqlite3.IntegrityError, match="lanzamiento tardio"):
        conn.execute(
            "INSERT INTO m3_lanzamientos_tardios (run_id, sku, causa) "
            "VALUES ('m3-a', 'S1', 'lanzado despues del corte')"
        )
    with pytest.raises(sqlite3.IntegrityError, match="lanzamiento tardio"):
        conn.execute("INSERT INTO m3_skus (run_id, sku) VALUES ('m3-a', 'TARDE')")
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        conn.execute(
            "INSERT INTO m3_lanzamientos_tardios (run_id, sku, causa) "
            "VALUES ('m3-a', 'OTRO', 'sin historia admisible')"
        )
    conn.close()


def test_observacion_exige_corte_y_fecha_de_la_reserva(tmp_path):
    conn = _base_version_9(tmp_path / "central.db")
    _publicada(conn, "m3-a", [("S1", "smooth", "VALIDADO", "classical")])
    conn.commit()
    aplicar_migraciones_m3(conn)
    insertar = (
        "INSERT INTO m3_observaciones (run_id, sku, fecha, valor) "
        "VALUES ('m3-a', 'S1', ?, ?)"
    )
    with pytest.raises(sqlite3.IntegrityError, match="corte de su corrida"):
        conn.execute(insertar, ("2024-01-15", 3.0))
    conn.execute("INSERT INTO m3_cortes (run_id, t_star) VALUES ('m3-a', '2024-01-14')")
    with pytest.raises(sqlite3.IntegrityError, match="fecha de la reserva"):
        conn.execute(insertar, ("2024-01-14", 3.0))
    conn.execute(insertar, ("2024-01-15", 3.0))
    conn.execute(insertar, ("2024-01-16", None))
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        conn.execute(
            "INSERT INTO m3_observaciones (run_id, sku, fecha, valor) "
            "VALUES ('m3-a', 'S9', '2024-01-15', 1.0)"
        )
    assert conn.execute(
        "SELECT fecha, valor FROM m3_observaciones ORDER BY fecha"
    ).fetchall() == [("2024-01-15", 3.0), ("2024-01-16", None)]
    conn.close()
