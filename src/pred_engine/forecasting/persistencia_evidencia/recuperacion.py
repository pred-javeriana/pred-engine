"""Protocolo congelado y lectura de unidades durables para la recuperacion 3.0."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from pred_engine.comun.exclusiones import SkuExclusion
from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos.manifiesto_candidatos import (
    VERSION_ESQUEMA_MANIFIESTO,
    Candidato,
    ManifiestoCandidatos,
)
from pred_engine.forecasting.adaptador_candidatos import FalloCandidato, HandoffValidado
from pred_engine.forecasting.evaluaciones.calculo_errores import (
    PronosticoFechado,
    SerieCandidato,
)
from pred_engine.forecasting.generador_pronosticos import (
    FalloPronostico,
    UnidadPronostico,
)
from pred_engine.forecasting.persistencia_evidencia.contratos import IdentidadCorrida
from pred_engine.forecasting.persistencia_evidencia.repositorio import (
    _exigir_estado,
    _fijar_corte,
    _inconsistente,
    _insertar_unidades,
    _transaccion,
)

_logger = get_logger(__name__)


def fijar_protocolo(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    protocolo: dict[str, Any],
    manifiesto: str,
    procedencia: dict[str, Any],
    reserva_abierta_en: datetime,
    n_unidades_previstas: int,
    t_star: pd.Timestamp,
    lanzamientos_tardios: Sequence[SkuExclusion],
    ejecucion_id: int | None = None,
) -> None:
    """Fija protocolo y handoff antes del primer pronostico; nunca los reemplaza.

    En la misma transaccion fija el corte ``t_star`` y los lanzamientos tardios
    de M1 que uso L2, que M3 publica para M4 (``m3_cortes``).
    """
    canonico = json.dumps(protocolo, sort_keys=True, separators=(",", ":"))
    # La emision del handoff se conserva. Reanudar no es volver a emitirlo.
    sobre = json.dumps(json.loads(manifiesto), sort_keys=True, separators=(",", ":"))
    with _transaccion(conn):
        estado = _exigir_estado(
            conn,
            run_id,
            ("EN_CURSO", "EVALUADA", "SELECCIONADA", "PUBLICADA"),
            "fijar_protocolo",
        )
        previa = conn.execute(
            "SELECT protocolo, manifiesto, n_unidades_previstas FROM m3_protocolos "
            "WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if previa is not None:
            if previa != (canonico, sobre, n_unidades_previstas):
                raise _inconsistente(
                    "protocolo o manifiesto incompatible con la corrida previa"
                )
        else:
            conn.execute(
                "INSERT INTO m3_protocolos "
                "(run_id, ejecucion_id, protocolo, manifiesto, procedencia, "
                "reserva_abierta_en, n_unidades_previstas) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    ejecucion_id,
                    canonico,
                    sobre,
                    json.dumps(procedencia, sort_keys=True),
                    reserva_abierta_en.isoformat(),
                    n_unidades_previstas,
                ),
            )
        _fijar_corte(conn, run_id, estado, t_star, lanzamientos_tardios)


def restaurar_handoff(
    conn: sqlite3.Connection, run_id: str, *, manifiesto_pendiente: str | None = None
) -> HandoffValidado | None:
    """Restaura la decision durable de 3.2 sin cargar pesos ni cambiar sus fallos."""
    guardado = protocolo_guardado(conn, run_id)
    if guardado is None:
        fila = conn.execute(
            "SELECT estado FROM m3_corridas WHERE run_id = ?", (run_id,)
        ).fetchone()
        if fila is None or fila[0] != "EN_CURSO" or manifiesto_pendiente is None:
            return None
        # Caida despues de registrar, pero antes de fijar el protocolo. Aun
        # no hay ventanas abiertas y los fallos de 3.2 ya son durables.
        texto = manifiesto_pendiente
    else:
        texto = guardado["manifiesto"]
    manifiesto = ManifiestoCandidatos.model_validate_json(texto)
    if manifiesto.schema_version != VERSION_ESQUEMA_MANIFIESTO:
        raise _inconsistente("version de handoff durable no soportada")
    filas = conn.execute(
        "SELECT sku, candidato_id, modelo, campo, codigo_error, mensaje FROM m3_fallos "
        "WHERE run_id = ? ORDER BY rowid",
        (run_id,),
    ).fetchall()
    fallos = tuple(FalloCandidato(*f) for f in filas)
    rechazados = {f.candidato_id for f in fallos}
    adaptador: TypeAdapter[Candidato] = TypeAdapter(Candidato)
    candidatos = tuple(
        adaptador.validate_python(c)
        for c in manifiesto.candidatos
        if c["candidato_id"] not in rechazados
    )
    return HandoffValidado(
        manifiesto.contexto, manifiesto.run_id_m2, candidatos, fallos
    )


def guardar_unidades_generadas(
    conn: sqlite3.Connection, run_id: str, unidades: Sequence[UnidadPronostico]
) -> int:
    """Resultados, fallos y marcadores de varias ventanas en una sola transaccion.

    Ninguna unidad queda durable sin las demas, y una ya confirmada no se toca.
    Devuelve cuantas eran nuevas.
    """
    if not unidades:
        return 0
    nuevas = 0
    with _transaccion(conn):
        _exigir_estado(conn, run_id, ("EN_CURSO",), "guardar_unidades_generadas")
        for unidad in unidades:
            c = unidad.candidato
            nuevas += _insertar_unidades(
                conn,
                run_id,
                c.sku,
                SerieCandidato(
                    c.candidato_id, c.familia, c.modelo, (unidad.pronostico,)
                ),
                codigo_version=unidad.codigo_version,
                semilla=c.semilla,
                emitido_en=unidad.emitido_en,
                fallos={}
                if unidad.fallo is None
                else {unidad.pronostico.origen: unidad.fallo},
            )
    _logger.info(
        "Unidades generadas run_id=%s nuevas=%d de %d", run_id, nuevas, len(unidades)
    )
    return nuevas


def cargar_serie(
    conn: sqlite3.Connection,
    run_id: str,
    sku: str,
    candidato_id: str,
    familia: str,
    modelo: str,
) -> SerieCandidato:
    """Lee pronosticos confirmados y fallidos en orden causal, sin recalcularlos."""
    origenes = conn.execute(
        "SELECT origen, estado, causa FROM m3_unidades WHERE run_id = ? AND sku = ? "
        "AND candidato_id = ? ORDER BY origen",
        (run_id, sku, candidato_id),
    ).fetchall()
    protocolo = protocolo_guardado(conn, run_id)
    horizonte = None
    if protocolo is not None:
        try:
            horizonte = protocolo["protocolo"]["pronosticos"]["horizonte"]
        except (KeyError, TypeError) as exc:
            raise _inconsistente("protocolo durable sin horizonte") from exc
        if type(horizonte) is not int or horizonte < 1:
            raise _inconsistente("horizonte invalido en el protocolo durable")
    pronosticos = []
    for origen, estado, causa in origenes:
        filas = conn.execute(
            "SELECT h, fecha, valor FROM m3_pronosticos WHERE run_id = ? AND sku = ? "
            "AND candidato_id = ? AND origen = ? ORDER BY h",
            (run_id, sku, candidato_id, origen),
        ).fetchall()
        if (
            not filas
            or (horizonte is not None and len(filas) != horizonte)
            or [f[0] for f in filas] != list(range(1, len(filas) + 1))
        ):
            raise _inconsistente("unidad durable incompleta o sin pronosticos")
        fechas = pd.DatetimeIndex([f[1] for f in filas])
        inicio = pd.Timestamp(origen)
        if not isinstance(inicio, pd.Timestamp):
            raise _inconsistente("origen invalido en la unidad durable")
        if not fechas.equals(
            pd.date_range(inicio + pd.Timedelta(days=1), periods=len(filas))
        ):
            raise _inconsistente("fechas incompatibles en la unidad durable")
        valores = np.array(
            [np.nan if f[2] is None else f[2] for f in filas], dtype=float
        )
        if (
            estado == "fallida"
            and (causa is None or np.isfinite(valores).all() or np.isinf(valores).any())
        ) or (
            estado == "confirmada"
            and (causa is not None or not np.isfinite(valores).all())
        ):
            raise _inconsistente("estado y valores incompatibles en la unidad durable")
        pronosticos.append(PronosticoFechado(inicio, fechas, valores))
    return SerieCandidato(candidato_id, familia, modelo, tuple(pronosticos))


def cargar_unidades(
    conn: sqlite3.Connection,
    run_id: str,
    candidato: Candidato,
) -> tuple[UnidadPronostico, ...]:
    """Reconstruye las unidades del candidato con su trazabilidad original."""
    serie = cargar_serie(
        conn,
        run_id,
        candidato.sku,
        candidato.candidato_id,
        candidato.familia,
        candidato.modelo,
    )
    unidades = []
    for p in serie.pronosticos:
        estado, causa, emision, version, semilla = conn.execute(
            "SELECT estado, causa, emitido_en, codigo_version, semilla "
            "FROM m3_unidades "
            "WHERE run_id = ? AND sku = ? AND candidato_id = ? AND origen = ?",
            (
                run_id,
                candidato.sku,
                candidato.candidato_id,
                p.origen.date().isoformat(),
            ),
        ).fetchone()
        if semilla != candidato.semilla or not emision or not version:
            raise _inconsistente("trazabilidad incompatible en la unidad durable")
        fallo = None
        if estado == "fallida":
            datos = json.loads(causa)
            datos["origen"] = pd.Timestamp(datos["origen"])
            fallo = FalloPronostico(**datos)
        elif causa is not None:
            raise _inconsistente("una unidad confirmada no puede tener causa de fallo")
        unidades.append(
            UnidadPronostico(
                candidato,
                version,
                datetime.fromisoformat(emision),
                p,
                fallo,
            )
        )
    return tuple(unidades)


def candidatos_con_fallos(
    conn: sqlite3.Connection, run_id: str, sku: str
) -> frozenset[str]:
    """Candidatos con al menos una ventana fallida; la linea base va aparte."""
    return frozenset(
        f[0]
        for f in conn.execute(
            "SELECT DISTINCT candidato_id FROM m3_unidades "
            "WHERE run_id = ? AND sku = ? "
            "AND estado = 'fallida' AND familia != 'seasonal_naive'",
            (run_id, sku),
        )
    )


def protocolo_de_ejecucion(
    conn: sqlite3.Connection, ejecucion_id: int
) -> tuple[IdentidadCorrida, dict[str, Any]] | None:
    """Busca la autoridad durable sin depender de archivos ni candidatos de M2."""
    filas = conn.execute(
        "SELECT c.run_id, c.identidad FROM m3_protocolos p "
        "JOIN m3_corridas c ON c.run_id = p.run_id WHERE p.ejecucion_id = ?",
        (ejecucion_id,),
    ).fetchall()
    if len(filas) > 1:
        raise _inconsistente("la ejecucion tiene varios protocolos M3 durables")
    if not filas:
        return None
    run_id, texto = filas[0]
    identidad = IdentidadCorrida.model_validate_json(texto)
    if identidad.run_id != run_id:
        raise _inconsistente("identidad incompatible con el protocolo durable")
    guardado = protocolo_guardado(conn, run_id)
    assert guardado is not None
    return identidad, guardado


def protocolo_guardado(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    fila = conn.execute(
        "SELECT protocolo, manifiesto, procedencia, reserva_abierta_en, "
        "n_unidades_previstas FROM m3_protocolos WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if fila is None:
        return None
    return {
        "protocolo": json.loads(fila[0]),
        "manifiesto": fila[1],
        "procedencia": json.loads(fila[2]),
        "reserva_abierta_en": fila[3],
        "n_unidades_previstas": fila[4],
    }
