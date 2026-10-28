"""Origen de datos: una ausencia no declara datos reales.

``origen_ingesta`` es el unico veredicto del origen real o sintetico de una
ingesta. Lo usan la ingesta M1, la corrida y ``pred-engine origen``, que la
plataforma consulta antes de encolar una corrida.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pred_engine.ingesta.informe import IngestReportFile


class OrigenIncompatibleError(ValueError):
    """Una declaracion contradice el origen registrado: provenance.origin_conflict."""


def resolver_origen(conocida: bool | None, declarada: bool | None) -> bool | None:
    """Conserva una declaración previa y rechaza su sustitución."""
    if any(
        value is not None and type(value) is not bool for value in (conocida, declarada)
    ):
        raise ValueError("el origen debe ser booleano o nulo")
    if conocida is not None and declarada is not None and conocida != declarada:
        raise OrigenIncompatibleError(
            "el origen real o sintetico de una ingesta no puede cambiar"
        )
    return conocida if declarada is None else declarada


def origen_ingesta(
    conn: sqlite3.Connection | None,
    *,
    data_root: str | Path,
    ingesta_sha256: str,
    declarada: bool | None,
    ingesta_ref_m1: str | None = None,
) -> bool | None:
    """Origen de la ingesta segun toda declaracion conocida, o ``None`` si no hay.

    Une las declaraciones de la ingesta y de toda otra ingesta que publico los
    mismos bytes Parquet (``ingesta_ref_m1``, su informe M1 o ``m3_ingestas``):
    ``m3_ingestas`` y ``m3_corridas`` con ``origen_declarado=1``, el informe M1
    canonico de la ingesta, los informes aceptados de esos mismos bytes y el
    ``datos_sinteticos`` de la configuracion de cada ejecucion. ``declarada`` se
    suma a ellas. Solo lee: sin ``conn`` consulta solo los informes M1.
    """
    if declarada is not None and type(declarada) is not bool:
        raise ValueError("el origen debe ser booleano o nulo")
    informes = _informes_m1(Path(data_root))
    columnas = _columnas_m3(conn)
    huellas = set() if ingesta_ref_m1 is None else {ingesta_ref_m1}
    propio = informes.get(ingesta_sha256)
    if (
        propio is not None
        and propio.published is not None
        and propio.published.sha256 is not None
    ):
        huellas.add(propio.published.sha256)
    if conn is not None and "m3_ingestas" in columnas:
        huellas.update(
            r[0]
            for r in conn.execute(
                "SELECT ingesta_ref_m1 FROM m3_ingestas WHERE ingesta_sha256 = ?",
                (ingesta_sha256,),
            )
        )
    mismos_bytes = {
        sha
        for sha, informe in informes.items()
        if informe.status == "accepted"
        and informe.published is not None
        and informe.published.sha256 in huellas
    }
    if conn is not None and "m3_ingestas" in columnas:
        for huella in huellas:
            mismos_bytes.update(
                r[0]
                for r in conn.execute(
                    "SELECT ingesta_sha256 FROM m3_ingestas WHERE ingesta_ref_m1 = ?",
                    (huella,),
                )
            )
    origenes: dict[bool, str] = {}
    if propio is not None and propio.source_file.datos_sinteticos is not None:
        origenes.setdefault(propio.source_file.datos_sinteticos, "su informe M1")
    for sha in sorted(mismos_bytes - {ingesta_sha256}):
        informe = informes.get(sha)
        if (
            informe is not None
            and informe.status == "accepted"
            and informe.source_file.datos_sinteticos is not None
        ):
            origenes.setdefault(
                informe.source_file.datos_sinteticos,
                f"la ingesta {sha} que publico los mismos bytes",
            )
    if conn is not None:
        for sha in sorted(mismos_bytes | {ingesta_sha256}):
            for valor, fuente in _declaradas_en_base(conn, sha, columnas):
                origenes.setdefault(valor, fuente)
    if declarada is not None:
        origenes.setdefault(declarada, "la declaracion recibida")
    if len(origenes) > 1:
        raise OrigenIncompatibleError(
            "el origen real o sintetico de una ingesta no puede cambiar: "
            f"{origenes[True]} la declara sintetica y {origenes[False]}, real"
        )
    return next(iter(origenes), None)


def _informes_m1(data_root: Path) -> dict[str, IngestReportFile]:
    """Informes M1 canonicos por ingesta; no crea el arbol de datos."""
    from pred_engine.ingesta.informe import IngestReportFile

    carpeta = data_root / "ingestas"
    informes = {}
    for path in sorted(carpeta.glob("[0-9a-f]" * 64 + ".json")):
        informe = IngestReportFile.model_validate_json(path.read_text())
        if informe.ingest_id != path.stem or informe.source_file.sha256 != path.stem:
            raise ValueError("la procedencia M1 tiene identidades incompatibles")
        informes[path.stem] = informe
    return informes


def _columnas_m3(conn: sqlite3.Connection | None) -> dict[str, set[str]]:
    if conn is None:
        return {}
    tablas = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    return {
        tabla: {r[1] for r in conn.execute(f"PRAGMA table_info({tabla})")}
        for tabla in ("m3_ingestas", "m3_corridas", "ejecuciones")
        if tabla in tablas
    }


def _declaradas_en_base(
    conn: sqlite3.Connection, sha: str, columnas: dict[str, set[str]]
) -> list[tuple[bool, str]]:
    declaradas = []
    for tabla, fuente in (("m3_ingestas", "M1"), ("m3_corridas", "una corrida M3")):
        if tabla not in columnas:
            continue
        filtro = (
            " AND origen_declarado = 1" if "origen_declarado" in columnas[tabla] else ""
        )
        declaradas.extend(
            (bool(r[0]), f"{fuente} de la ingesta {sha}")
            for r in conn.execute(
                f"SELECT datos_sinteticos FROM {tabla} "
                f"WHERE ingesta_sha256 = ?{filtro}",
                (sha,),
            )
        )
    if "ejecuciones" in columnas:
        for (texto,) in conn.execute(
            "SELECT c.parametros FROM configuraciones c "
            "JOIN ejecuciones e ON e.configuracion_id = c.id "
            "JOIN ingestas i ON i.id = e.ingesta_id WHERE i.sha256 = ?",
            (sha,),
        ):
            parametros = json.loads(texto)
            if not isinstance(parametros, dict):
                raise ValueError("procedencia invalida en la configuracion almacenada")
            valor = parametros.get("datos_sinteticos")
            if valor is not None and type(valor) is not bool:
                raise ValueError("procedencia invalida en la configuracion almacenada")
            if valor is not None:
                declaradas.append(
                    (valor, f"la configuracion de una ejecucion de la ingesta {sha}")
                )
    return declaradas
