"""Identidad M1/M2 y estado global sobre la base unica de plataforma."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pred_engine.comun.procedencia import resolver_origen
from pred_engine.forecasting.persistencia_evidencia.repositorio import _transaccion

if TYPE_CHECKING:
    from pred_engine.pipeline import IngestionArtifact

# Ajustes que cambian resultados y que toda ejecucion declara, en la plataforma
# y en el motor. Los ajustes de ejecucion (workers) y los sobres propios de cada
# representacion no cuentan. Una configuracion antigua de la plataforma sin
# familias ni ensayos uso los valores por omision del CLI.
AJUSTES_RESULTADO = (
    "families",
    "trials",
    "min_train",
    "horizon",
    "step",
    "seasonality",
    "metric",
    "seed",
)
_POR_OMISION = {"families": ["classical", "ml", "dl"], "trials": None}
# Opciones que solo el CLI registra; se comparan cuando ambas partes las tienen.
AJUSTES_OPCIONALES = ("provider", "model", "dm_min_origins", "code")


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


def origen_huella(conn: sqlite3.Connection, ingesta_ref_m1: str) -> bool | None:
    """Origen declarado de unos bytes Parquet, sea cual sea su ingesta."""
    origenes = {
        bool(r[0])
        for r in conn.execute(
            "SELECT datos_sinteticos FROM m3_ingestas "
            "WHERE ingesta_ref_m1 = ? AND origen_declarado = 1",
            (ingesta_ref_m1,),
        )
    }
    if len(origenes) > 1:
        raise ValueError("la ingesta tiene origenes incompatibles")
    return next(iter(origenes), None)


def buscar_ingesta(
    conn: sqlite3.Connection, parquet: Path, data_root: Path, ingesta_ref_m1: str
) -> tuple[str, bool | None] | None:
    """Reusa el CSV original; las rutas relativas pertenecen a data_root.

    El origen ya declarado para esos mismos bytes se conserva aunque varias
    ingestas compartan la huella o ninguna se identifique por su ruta.
    """
    por_bytes = origen_huella(conn, ingesta_ref_m1)
    previa = _buscar_ingesta(conn, parquet, data_root, ingesta_ref_m1)
    if previa is None:
        return None if por_bytes is None else (ingesta_ref_m1, por_bytes)
    return previa[0], resolver_origen(previa[1], por_bytes)


def _buscar_ingesta(
    conn: sqlite3.Connection, parquet: Path, data_root: Path, ingesta_ref_m1: str
) -> tuple[str, bool | None] | None:
    exactas = conn.execute(
        "SELECT ingesta_sha256, datos_sinteticos, origen_declarado FROM m3_ingestas "
        "WHERE ingesta_ref_m1 = ?",
        (ingesta_ref_m1,),
    ).fetchall()
    if len(exactas) == 1:
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
    if sha is not None and not _verificar_contenido(
        conn, sha, data_root, ingesta_ref_m1, destino
    ):
        # Sin ninguna huella, la ruta no prueba que estos bytes sean esa ingesta.
        sha = None
    conocida = origen_registrado(conn, sha) if sha is not None else None
    if sha is not None and conocida is not None:
        return sha, conocida
    from pred_engine.ingesta.informe import IngestReportFile, son_informes_equivalentes

    coincidentes: list[IngestReportFile] = []
    # Un archivo ajeno no es un informe M1. Si ya se conoce la fuente, solo
    # su informe canonico puede aportar procedencia y se valida estrictamente.
    patron = f"{sha}.json" if sha is not None else "[0-9a-f]" * 64 + ".json"
    for path in (data_root / "ingestas").glob(patron):
        report = IngestReportFile.model_validate_json(path.read_text())
        if report.status != "accepted":
            continue
        publicaciones = (
            [report.published] if report.published is not None else []
        ) + report.publications
        if any(p is not None and p.sha256 == ingesta_ref_m1 for p in publicaciones):
            if report.ingest_id != report.source_file.sha256:
                raise ValueError("la procedencia M1 tiene identidades incompatibles")
            coincidentes.append(report)
    if len(coincidentes) > 1:
        if all(
            son_informes_equivalentes(coincidentes[0], otro)
            for otro in coincidentes[1:]
        ):
            canonical = min(coincidentes, key=lambda r: r.ingest_id)
            declarada = (canonical.ingest_id, canonical.source_file.datos_sinteticos)
            if sha is not None and sha not in {r.ingest_id for r in coincidentes}:
                raise ValueError(
                    "la referencia de plataforma difiere de la procedencia M1"
                )
            return declarada
        raise ValueError("el Parquet tiene varias procedencias M1 declaradas")
    if coincidentes:
        report = coincidentes[0]
        declarada = (report.ingest_id, report.source_file.datos_sinteticos)
        if sha is not None and sha != declarada[0]:
            raise ValueError("la referencia de plataforma difiere de la procedencia M1")
        return declarada
    return (sha, None) if sha is not None else None


def origen_conocido(
    conn: sqlite3.Connection,
    *,
    fuente_sha256: str | None,
    parquet: Path,
    data_root: Path,
    ingesta_ref_m1: str,
) -> tuple[str, bool | None]:
    """Fuente M1 y el origen que ya declararon la base o su informe canonico."""
    from pred_engine.ingesta.informe import origen_informe

    if fuente_sha256 is None:
        previa = buscar_ingesta(conn, parquet, data_root, ingesta_ref_m1)
        return previa if previa is not None else (ingesta_ref_m1, None)
    conocida = resolver_origen(
        origen_registrado(conn, fuente_sha256), origen_informe(data_root, fuente_sha256)
    )
    return fuente_sha256, resolver_origen(conocida, origen_huella(conn, ingesta_ref_m1))


def _ajustes(parametros: Mapping[str, Any]) -> dict[str, Any]:
    completos = {**_POR_OMISION, **parametros}
    return {
        k: json.loads(json.dumps(completos[k]))
        for k in (*AJUSTES_RESULTADO, *AJUSTES_OPCIONALES)
        if k in completos
    }


def _ajustes_distintos(
    guardados: Mapping[str, Any], nuevos: Mapping[str, Any]
) -> list[str]:
    """Un ajuste obligatorio ausente en una de las partes tambien es distinto."""
    a, b = _ajustes(guardados), _ajustes(nuevos)
    comunes = (k for k in AJUSTES_OPCIONALES if k in a and k in b)
    return sorted(
        k
        for k in (*AJUSTES_RESULTADO, *comunes)
        if k not in a or k not in b or a[k] != b[k]
    )


def _verificar_contenido(
    conn: sqlite3.Connection, sha: str, data_root: Path, ingesta_ref_m1: str, ruta: Path
) -> bool:
    """Una ruta registrada solo identifica la ingesta si conserva su contenido.

    Devuelve si existe alguna huella con la que comparar.
    """
    from pred_engine.ingesta.informe import IngestReportFile, report_path

    huellas = {
        r[0]
        for r in conn.execute(
            "SELECT ingesta_ref_m1 FROM m3_ingestas WHERE ingesta_sha256 = ?", (sha,)
        )
    }
    informe = report_path(data_root, sha)
    if informe.exists():
        report = IngestReportFile.model_validate_json(informe.read_text())
        if report.published is not None and report.published.sha256 is not None:
            huellas.add(report.published.sha256)
    if huellas and ingesta_ref_m1 not in huellas:
        raise ValueError(
            f"el Parquet registrado en {ruta} cambio de contenido: la procedencia "
            "de su ingesta ya no aplica; vuelva a clasificar el archivo"
        )
    return bool(huellas)


def asegurar_referencias(
    conn: sqlite3.Connection,
    ingesta: IngestionArtifact,
    *,
    ingesta_ref_m1: str,
    run_id_m2: str,
    directorio: Path,
    configuracion: Mapping[str, Any],
    data_root: Path,
    datos_sinteticos: bool | None,
) -> tuple[str, int, bool | None]:
    """Conserva las referencias; nunca declara terminada una evaluacion pendiente.

    Devuelve el origen resuelto: la declaracion del llamador no sustituye el
    origen que M1 ya registro, y un conflicto se rechaza antes de evaluar.
    """
    panel = ingesta.panel
    sha, conocida = origen_conocido(
        conn,
        fuente_sha256=ingesta.source_sha256,
        parquet=ingesta.parquet_path,
        data_root=data_root,
        ingesta_ref_m1=ingesta_ref_m1,
    )
    # La entrada, M1 o la base y el validador deben coincidir cuando declaran.
    origen = resolver_origen(
        resolver_origen(conocida, ingesta.datos_sinteticos), datos_sinteticos
    )
    if "seed" not in configuracion:
        raise ValueError("la configuracion M2 no declara la semilla que uso")
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
            "SELECT e.id, e.ingesta_id, c.parametros, e.seed FROM ejecuciones e "
            "JOIN configuraciones c ON c.id = e.configuracion_id WHERE e.run_id = ?",
            (run_id_m2,),
        ).fetchone()
        if existente is not None:
            if existente[1] != ingesta_id:
                raise ValueError("run_id de M2 pertenece a otra ingesta de plataforma")
            # La semilla de la ejecucion es la de su columna, comun a ambos sistemas.
            distintos = _ajustes_distintos(
                {**json.loads(existente[2]), "seed": existente[3]}, configuracion
            )
            if distintos:
                raise ValueError(
                    "run_id de M2 pertenece a otra configuracion de resultados: "
                    + ", ".join(distintos)
                )
            return sha, existente[0], origen
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
                configuracion["seed"],
                run_id_m2,
                str(directorio),
            ),
        ).lastrowid
    assert ejecucion_id is not None
    return sha, ejecucion_id, origen


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
