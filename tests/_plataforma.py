"""Frontera minima de la plataforma para pruebas unitarias del motor.

La prueba coordinada usa las migraciones reales de pred-platform. Esta base
solo declara las tablas previas que referencia M3, sin importar la plataforma.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def crear_base_plataforma(ruta: Path) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ruta)
    conn.executescript("""
        CREATE TABLE ingestas (
            id INTEGER PRIMARY KEY, sha256 TEXT UNIQUE NOT NULL,
            nombre_archivo TEXT NOT NULL, filas INTEGER NOT NULL, skus INTEGER NOT NULL,
            ruta_parquet TEXT
        );
        CREATE TABLE configuraciones (
            id INTEGER PRIMARY KEY, version INTEGER NOT NULL,
            descripcion TEXT, parametros TEXT NOT NULL
        );
        CREATE TABLE ejecuciones (
            id INTEGER PRIMARY KEY, ingesta_id INTEGER NOT NULL REFERENCES ingestas(id),
            configuracion_id INTEGER NOT NULL REFERENCES configuraciones(id),
            seed INTEGER NOT NULL, estado TEXT NOT NULL,
            run_id TEXT UNIQUE, ruta_corrida TEXT
        );
        PRAGMA user_version = 4;
    """)
    conn.close()
    return ruta
