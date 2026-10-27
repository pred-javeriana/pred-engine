"""Migraciones M3 compartidas con la base unica de la plataforma (ADR-03-009)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path

from pred_engine.comun.logger import get_logger

_logger = get_logger(__name__)

# Las versiones 1 a 4 pertenecen a la plataforma. Estos textos son la fuente
# unica de sus migraciones 0005 a 0009; tambien los aplica el CLI del motor.
MIGRACIONES_M3 = (
    (5, "esquema_m3.sql"),
    (6, "esquema_m3_0002.sql"),
    (7, "esquema_m3_0003.sql"),
    (8, "esquema_m3_0004.sql"),
    (9, "esquema_m3_0005.sql"),
)
VERSION_ESQUEMA_M3 = MIGRACIONES_M3[-1][0]
# Ultima version de la base unica que el motor conoce. La 10 es de la plataforma
# (estado finalizo_m2 de las tareas): el motor la acepta, pero no la aplica.
VERSION_PLATAFORMA = 10


class BasePlataformaError(ValueError):
    """No existe la base unica o no tiene el esquema previo de la plataforma."""


def textos_migraciones_m3() -> tuple[tuple[int, str], ...]:
    paquete = files("pred_engine.forecasting.persistencia_evidencia")
    return tuple(
        (version, paquete.joinpath(nombre).read_text(encoding="utf-8"))
        for version, nombre in MIGRACIONES_M3
    )


def _sentencias(texto: str) -> Iterator[str]:
    pendiente = ""
    for linea in texto.splitlines(keepends=True):
        pendiente += linea
        if sqlite3.complete_statement(pendiente):
            yield pendiente
            pendiente = ""
    if pendiente.strip():
        raise ValueError("migracion M3 incompleta")


def aplicar_migraciones_m3(conn: sqlite3.Connection) -> tuple[int, ...]:
    """Aplica solo las migraciones M3 pendientes, con la version y el DDL atomicos.

    No crea la base ni las tablas de M1/M2. Comparte ``PRAGMA user_version``
    con la plataforma y exige sus migraciones previas (1 a 4). Una conexion
    con transaccion abierta se rechaza sin confirmar trabajo ajeno.
    """
    if conn.in_transaction:
        raise BasePlataformaError("la conexion tiene una transaccion abierta")
    conn.execute("PRAGMA foreign_keys = ON")
    aplicadas: list[int] = []
    for version, texto in textos_migraciones_m3():
        conn.execute("BEGIN IMMEDIATE")
        try:
            actual = conn.execute("PRAGMA user_version").fetchone()[0]
            if actual < 4 or actual > VERSION_PLATAFORMA:
                raise BasePlataformaError(
                    f"base de plataforma con version {actual}; se requiere entre 4 y "
                    f"{VERSION_PLATAFORMA}. Ejecute sus migraciones de plataforma"
                )
            if actual < version:
                for sentencia in _sentencias(texto):
                    conn.execute(sentencia)
                violaciones = conn.execute("PRAGMA foreign_key_check").fetchall()
                if violaciones:
                    raise BasePlataformaError("la base contiene referencias huerfanas")
                conn.execute(f"PRAGMA user_version = {version}")
                aplicadas.append(version)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    if aplicadas:
        _logger.info("Migraciones M3 aplicadas versiones=%s", aplicadas)
    return tuple(aplicadas)


@contextmanager
def abrir_base_plataforma(ruta: str | Path) -> Iterator[sqlite3.Connection]:
    """Abre una base existente en modo rw, nunca crea otra base de resultados."""
    path = Path(ruta).expanduser().resolve()
    try:
        conn = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=30)
    except sqlite3.OperationalError as exc:
        raise BasePlataformaError(
            f"no existe o no se puede abrir la base de la plataforma: {path}; "
            "creela primero mediante la migracion de pred-platform"
        ) from exc
    try:
        aplicar_migraciones_m3(conn)
        yield conn
    finally:
        conn.close()
