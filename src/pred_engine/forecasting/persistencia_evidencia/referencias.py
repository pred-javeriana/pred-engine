"""Identidad M1/M2 y estado global sobre la base unica de plataforma."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pred_engine.forecasting.persistencia_evidencia.repositorio import _transaccion

if TYPE_CHECKING:
    from pred_engine.pipeline import IngestionArtifact


def origen_registrado(conn: sqlite3.Connection, sha: str) -> bool | None:
    """Solo una declaracion conocida puede fijar el origen de una ingesta."""
    origenes = {
        bool(r[0])
        for r in conn.execute(
            "SELECT datos_sinteticos FROM m3_ingestas "
            "WHERE ingesta_sha256=?1 AND origen_declarado=1 "
            "UNION SELECT datos_sinteticos FROM m3_corridas "
            "WHERE ingesta_sha256=?1 AND origen_declarado=1",
            (sha,),
        )
    }
    if len(origenes) > 1:
        raise ValueError("la ingesta tiene origenes incompatibles")
    return next(iter(origenes), None)


def buscar_ingesta(
    conn: sqlite3.Connection, parquet: Path, data_root: Path, ingesta_ref_m1: str
) -> tuple[str, bool | None] | None:
    """Reusa el CSV original; las rutas relativas pertenecen a data_root."""
    exactas = conn.execute(
        "SELECT ingesta_sha256, datos_sinteticos, origen_declarado FROM m3_ingestas "
        "WHERE ingesta_ref_m1 = ?",
        (ingesta_ref_m1,),
    ).fetchall()
    if len(exactas) == 1:
        from pred_engine.comun.procedencia import resolver_origen
        from pred_engine.ingesta.informe import origen_informe

        sha = exactas[0][0]
        conocida = origen_registrado(conn, sha)
        return sha, resolver_origen(conocida, origen_informe(data_root, sha))
    destino = parquet.expanduser().resolve()
    candidatas = set()
    for sha, ruta in conn.execute(
        "SELECT sha256, ruta_parquet FROM ingestas WHERE ruta_parquet IS NOT NULL"
    ):
        path = Path(ruta).expanduser()
        if not path.is_absolute():
            path = data_root / path
        if path.resolve() == destino:
            candidatas.add(sha)
    if len(candidatas) > 1:
        raise ValueError("la ruta Parquet corresponde a varias ingestas sin huella M1")
    sha = next(iter(candidatas), None)
    conocida = origen_registrado(conn, sha) if sha is not None else None
    if sha is not None and conocida is not None:
        return sha, conocida
    from pred_engine.ingesta.informe import IngestReportFile

    declaradas = set()
    # Un archivo ajeno no es un informe M1. Si ya se conoce la fuente, solo
    # su informe canonico puede aportar procedencia y se valida estrictamente.
    patron = f"{sha}.json" if sha is not None else "[0-9a-f]" * 64 + ".json"
    for path in (data_root / "ingestas").glob(patron):
        report = IngestReportFile.model_validate_json(path.read_text())
        if (
            report.status == "accepted"
            and report.published is not None
            and report.published.sha256 == ingesta_ref_m1
        ):
            if report.ingest_id != report.source_file.sha256:
                raise ValueError("la procedencia M1 tiene identidades incompatibles")
            declaradas.add((report.ingest_id, report.source_file.datos_sinteticos))
    if len(declaradas) > 1:
        raise ValueError("el Parquet tiene varias procedencias M1 declaradas")
    if declaradas:
        declarada = declaradas.pop()
        if sha is not None and sha != declarada[0]:
            raise ValueError("la referencia de plataforma difiere de la procedencia M1")
        return declarada
    return (sha, None) if sha is not None else None


def asegurar_referencias(
    conn: sqlite3.Connection,
    ingesta: IngestionArtifact,
    *,
    ingesta_ref_m1: str,
    run_id_m2: str,
    directorio: Path,
    configuracion: Mapping[str, Any],
    data_root: Path,
) -> tuple[str, int]:
    """Conserva las referencias; nunca declara terminada una evaluacion pendiente."""
    panel = ingesta.panel
    previa = (
        buscar_ingesta(conn, ingesta.parquet_path, data_root, ingesta_ref_m1)
        if ingesta.source_sha256 is None
        else None
    )
    sha = ingesta.source_sha256 or (previa[0] if previa else ingesta_ref_m1)
    with _transaccion(conn):
        conn.execute(
            "INSERT INTO ingestas (sha256, nombre_archivo, filas, skus, ruta_parquet) "
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
            "VALUES (?, ?, ?, 'ejecutando', ?, ?)",
            (
                ingesta_id,
                configuracion_id,
                configuracion.get("seed", 0),
                run_id_m2,
                str(directorio),
            ),
        ).lastrowid
    assert ejecucion_id is not None
    return sha, ejecucion_id


def finalizar_referencia(
    conn: sqlite3.Connection,
    ejecucion_id: int,
    *,
    hay_fallos: bool = False,
    error: Exception | None = None,
    conservar_completada: bool = False,
) -> None:
    """Registra el resultado L4 sin borrar su error ni confundir M2 con la corrida.

    ``conservar_completada`` deja intacta una ejecucion ya completada: reutilizar
    una publicacion congelada no reescribe el estado que describe esa corrida.
    """
    from pred_engine.ingesta.informe import ErrorDetail, error_code

    estado = (
        "detenida"
        if error is not None
        else ("completada_con_fallos" if hay_fallos else "completada")
    )
    detalle = (
        None
        if error is None
        else ErrorDetail(
            code=error_code(error),
            stage="L4",
            detail=str(error),
            source_exception=type(error).__name__,
        ).model_dump_json()
    )
    with _transaccion(conn):
        conn.execute(
            "UPDATE ejecuciones SET estado = ?, error_json = ? WHERE id = ? AND NOT "
            "(? AND estado IN ('completada', 'completada_con_fallos'))",
            (estado, detalle, ejecucion_id, conservar_completada),
        )
