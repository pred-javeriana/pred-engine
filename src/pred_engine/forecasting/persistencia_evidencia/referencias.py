"""Referencias M1/M2 de una corrida sobre la base unica de plataforma."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pred_engine.forecasting.persistencia_evidencia.repositorio import _transaccion

if TYPE_CHECKING:
    from pred_engine.pipeline import IngestionArtifact


def asegurar_referencias(
    conn: sqlite3.Connection,
    ingesta: IngestionArtifact,
    *,
    ingesta_ref_m1: str,
    run_id_m2: str,
    directorio: Path,
    configuracion: Mapping[str, Any],
    hay_fallos: bool,
) -> tuple[str, int]:
    """Conserva las referencias existentes o registra la ejecucion del CLI.

    El CSV aporta su identidad original. Para un Parquet se recupera primero
    la ingesta que ya conoce la plataforma. Si es un artefacto externo, ese
    archivo de entrada identifica la ingesta. Nunca crea otra base de datos.
    """
    panel = ingesta.panel
    previa = conn.execute(
        "SELECT sha256 FROM ingestas WHERE ruta_parquet = ?",
        (str(ingesta.parquet_path),),
    ).fetchone()
    sha = ingesta.source_sha256 or (previa[0] if previa else ingesta_ref_m1)
    with _transaccion(conn):
        conn.execute(
            "INSERT INTO ingestas "
            "(sha256, nombre_archivo, filas, skus, ruta_parquet) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT (sha256) DO NOTHING",
            (
                sha,
                ingesta.parquet_path.name,
                len(panel),
                panel["sku_id"].nunique(),
                str(ingesta.parquet_path),
            ),
        )
        ingesta_id = conn.execute(
            "SELECT id FROM ingestas WHERE sha256 = ?", (sha,)
        ).fetchone()[0]
        existente = conn.execute(
            "SELECT id, ingesta_id FROM ejecuciones WHERE run_id = ?", (run_id_m2,)
        ).fetchone()
        if existente is not None:
            if existente[1] != ingesta_id:
                raise ValueError("run_id de M2 pertenece a otra ingesta de plataforma")
            return sha, existente[0]
        configuracion_id = conn.execute(
            "INSERT INTO configuraciones (version, descripcion, parametros) "
            "VALUES (1, ?, ?)",
            ("Corrida M2 del motor", json.dumps(dict(configuracion), sort_keys=True)),
        ).lastrowid
        ejecucion_id = conn.execute(
            "INSERT INTO ejecuciones "
            "(ingesta_id, configuracion_id, seed, estado, run_id, ruta_corrida) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                ingesta_id,
                configuracion_id,
                configuracion.get("seed", 0),
                "completada_con_fallos" if hay_fallos else "completada",
                run_id_m2,
                str(directorio),
            ),
        ).lastrowid
    assert ejecucion_id is not None
    return sha, ejecucion_id
