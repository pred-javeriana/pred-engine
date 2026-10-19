"""La actualizacion M3 conserva la base unica y nunca la crea por su cuenta."""

import sqlite3
from pathlib import Path

import pytest
from tests._plataforma import crear_base_plataforma

from pred_engine.forecasting.persistencia_evidencia import (
    BasePlataformaError,
    abrir_base_plataforma,
    aplicar_migraciones_m3,
    esquema_m3,
)


def test_migracion_desde_plataforma_y_repeticion_sin_duplicados(tmp_path: Path):
    ruta = crear_base_plataforma(tmp_path / "central.db")
    with abrir_base_plataforma(ruta) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 6
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
    assert aplicar_migraciones_m3(conn) == (6,)
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
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 6


@pytest.mark.parametrize("version", [0, 3, 7])
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
