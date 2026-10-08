"""Escritura idempotente, publicacion y consulta de la evidencia de M3 (3.5-A1).

ADR-03-009: la evidencia vive en las tablas `m3_*` de la base de la plataforma
(`esquema_m3.sql`, su migracion 0002) y el Parquet solo se deriva al publicar.
ADR-03-010: claves naturales, una unidad confirmada no se reescribe y cada
transicion de estado va en la misma transaccion que el trabajo que la completa.
La conexion la abre quien orquesta; aqui no se crean tablas.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Collection, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pred_engine.comun.logger import get_logger
from pred_engine.forecasting.adaptador_candidatos import (
    HandoffValidado,
    LoteRechazadoError,
)
from pred_engine.forecasting.evaluaciones.calculo_errores import (
    EvaluacionSku,
    Metricas,
    PronosticoFechado,
    SerieCandidato,
)
from pred_engine.forecasting.evaluaciones.veredictos import ResultadoEvaluacion
from pred_engine.forecasting.persistencia_evidencia.contratos import (
    EstadoCorrida,
    IdentidadCorrida,
)
from pred_engine.forecasting.persistencia_evidencia.errores import (
    EstadoCorridaError,
    EvidenciaInconsistenteError,
    PersistenciaEvidenciaError,
)

_logger = get_logger(__name__)

VENTANA_AGREGADA = "agregada"
TABLAS_POR_SKU = (
    "m3_skus",
    "m3_fallos",
    "m3_unidades",
    "m3_pronosticos",
    "m3_metricas",
    "m3_veredictos",
)
TABLAS = ("m3_corridas", "m3_categorias", *TABLAS_POR_SKU)

_METRICAS = ("n_pares", "mae", "rmse", "me", "mase", "razon_sn")
_CONTEOS = ("n_ventanas_totales", "n_ventanas_validas", "n_pronosticos_validos")
# Tipo declarado en el DDL -> dtype: una columna toda NULL conserva su tipo.
_DTYPES = {"INTEGER": "Int64", "REAL": "Float64", "TEXT": "string"}


def esquema_m3() -> str:
    """DDL de las tablas `m3_*`; la plataforma lo aplica como su migracion 0002."""
    paquete = files("pred_engine.forecasting.persistencia_evidencia")
    return paquete.joinpath("esquema_m3.sql").read_text(encoding="utf-8")


def registrar_corrida(
    conn: sqlite3.Connection,
    identidad: IdentidadCorrida,
    handoff: HandoffValidado | LoteRechazadoError,
    *,
    ingesta_sha256: str,
    datos_sinteticos: bool,
) -> str:
    """Registra la corrida con el resultado de 3.2 y devuelve su `run_id`.

    Un lote rechazado queda en RECHAZADA con su causa; uno valido, en EN_CURSO
    con los SKUs de su manifiesto y sus fallos de candidato. Registrar otra vez
    la misma identidad no cambia nada. `ingesta_sha256` es la clave de
    `ingestas` (archivo fuente de M1); la identidad lleva `ingesta_ref_m1`, que
    es el Parquet publicado.
    """
    if isinstance(handoff, LoteRechazadoError):
        estado, causa, run_id_m2, fallos, skus = (
            "RECHAZADA",
            f"{handoff.motivo}: {handoff.detalle}",
            None,
            (),
            set(),
        )
    else:
        estado, causa, run_id_m2, fallos = (
            "EN_CURSO",
            None,
            handoff.run_id_m2,
            handoff.fallos,
        )
        skus = {c.sku for c in handoff.candidatos} | {f.sku for f in fallos}
    run_id = identidad.run_id
    with _transaccion(conn):
        nueva = conn.execute(
            "INSERT INTO m3_corridas (run_id, ingesta_sha256, run_id_m2, identidad,"
            " datos_sinteticos, estado, causa_rechazo) VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (run_id) DO NOTHING",
            (
                run_id,
                ingesta_sha256,
                run_id_m2,
                identidad.canonica(),
                datos_sinteticos,
                estado,
                causa,
            ),
        ).rowcount
        if nueva:
            conn.executemany(
                "INSERT INTO m3_skus (run_id, sku) VALUES (?, ?)",
                [(run_id, sku) for sku in sorted(skus)],
            )
            conn.executemany(
                "INSERT INTO m3_fallos (run_id, candidato_id, sku, modelo, campo,"
                " codigo_error, mensaje) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        run_id,
                        f.candidato_id,
                        f.sku,
                        f.modelo,
                        f.campo,
                        f.codigo_error,
                        f.mensaje,
                    )
                    for f in fallos
                ],
            )
    if nueva:
        _logger.info(
            "Corrida registrada run_id=%s estado=%s skus=%d fallos=%d",
            run_id,
            estado,
            len(skus),
            len(fallos),
        )
    else:
        _logger.info("Corrida ya registrada run_id=%s; sin cambios", run_id)
    return run_id


def guardar_unidades(
    conn: sqlite3.Connection, run_id: str, sku: str, serie: SerieCandidato
) -> int:
    """Confirma las ventanas de un candidato y devuelve cuantas eran nuevas.

    Cada origen es una unidad (SKU x candidato x ventana). Su marcador y sus
    pronosticos van en la misma transaccion; una unidad ya confirmada no se
    toca aunque llegue con otros valores.
    """
    nuevas = 0
    with _transaccion(conn):
        _exigir_estado(conn, run_id, ("EN_CURSO",), "guardar_unidades")
        for pronostico in serie.pronosticos:
            origen = _fecha(pronostico.origen)
            confirmada = conn.execute(
                "INSERT INTO m3_unidades (run_id, sku, candidato_id, origen, familia,"
                " modelo) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                (run_id, sku, serie.candidato_id, origen, serie.familia, serie.modelo),
            ).rowcount
            if not confirmada:
                continue
            nuevas += 1
            conn.executemany(
                "INSERT INTO m3_pronosticos (run_id, sku, candidato_id, origen, h,"
                " fecha, valor) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (run_id, sku, serie.candidato_id, origen, h, _fecha(fecha), valor)
                    for h, (fecha, valor) in enumerate(
                        zip(pronostico.fechas, _valores(pronostico), strict=True),
                        start=1,
                    )
                ],
            )
    _logger.info(
        "Unidades run_id=%s sku=%s candidato_id=%s nuevas=%d de %d",
        run_id,
        sku,
        serie.candidato_id,
        nuevas,
        len(serie.pronosticos),
    )
    return nuevas


def unidades_confirmadas(
    conn: sqlite3.Connection, run_id: str
) -> frozenset[tuple[str, str, pd.Timestamp]]:
    """`(sku, candidato_id, origen)` ya confirmadas: al reanudar no se recalculan."""
    filas = conn.execute(
        "SELECT sku, candidato_id, origen FROM m3_unidades WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    return frozenset(
        (sku, cid, pd.Timestamp.fromisoformat(origen)) for sku, cid, origen in filas
    )


def marcar_evaluada(conn: sqlite3.Connection, run_id: str) -> None:
    """EN_CURSO -> EVALUADA: quien orquesta declara confirmadas todas las unidades."""
    with _transaccion(conn):
        estado = _exigir_estado(
            conn, run_id, ("EN_CURSO", "EVALUADA"), "marcar_evaluada"
        )
        if estado == "EN_CURSO":
            _cambiar_estado(conn, run_id, "EVALUADA")


def guardar_evaluacion(
    conn: sqlite3.Connection,
    run_id: str,
    evaluaciones: Sequence[EvaluacionSku],
    resultado: ResultadoEvaluacion,
) -> None:
    """Metricas, seleccion y veredictos de 3.4: EVALUADA -> SELECCIONADA.

    Todo en una transaccion; en una corrida ya SELECCIONADA no escribe nada.
    Las versiones de politica y metricas deben ser las de la identidad, el
    origen de los datos el de la corrida, y cada modelo evaluado con ventanas,
    incluida la linea base, debe tener sus pronosticos confirmados.
    """
    with _transaccion(conn):
        estado = _exigir_estado(
            conn, run_id, ("EVALUADA", "SELECCIONADA"), "guardar_evaluacion"
        )
        if estado == "SELECCIONADA":
            _logger.info("Evaluacion ya guardada run_id=%s; sin cambios", run_id)
            return
        texto, datos_sinteticos = conn.execute(
            "SELECT identidad, datos_sinteticos FROM m3_corridas WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        identidad = json.loads(texto)
        declaradas = (identidad["version_politica"], identidad["version_metricas"])
        recibidas = (resultado.version_politica, resultado.version_metricas)
        if recibidas != declaradas:
            raise _inconsistente(
                f"corrida {run_id}: versiones {recibidas} distintas de las de su "
                f"identidad {declaradas}"
            )
        # ADR-03-008: con datos sinteticos no hay VALIDADO ni Diebold-Mariano.
        if resultado.datos_sinteticos != bool(datos_sinteticos):
            raise _inconsistente(
                f"corrida {run_id}: veredictos emitidos con datos_sinteticos="
                f"{resultado.datos_sinteticos}, distinto del de la corrida"
            )
        confirmados = set(
            conn.execute(
                "SELECT DISTINCT sku, candidato_id FROM m3_unidades WHERE run_id = ?",
                (run_id,),
            )
        )
        sin_pronosticos = sorted(
            (e.sku, c.candidato_id)
            for e in evaluaciones
            for c in (e.linea_base, *e.candidatos)
            if c.n_ventanas_totales and (e.sku, c.candidato_id) not in confirmados
        )
        if sin_pronosticos:
            raise _inconsistente(
                f"corrida {run_id}: {len(sin_pronosticos)} modelos evaluados sin "
                f"pronosticos confirmados, por ejemplo {sin_pronosticos[0]}"
            )
        conn.executemany(
            "INSERT INTO m3_metricas (run_id, sku, candidato_id, ventana, metrica,"
            " valor, causa) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [fila for e in evaluaciones for fila in _filas_metricas(run_id, e)],
        )
        conn.executemany(
            "INSERT INTO m3_categorias (run_id, sku_class, familia_campeona, motivo,"
            " adverso, medianas_r, skus_comparables, excluidos, familias_excluidas,"
            " conteos, porcentajes, mediana_r, n_adversos)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    run_id,
                    c.sku_class,
                    c.seleccion.familia_campeona,
                    c.seleccion.motivo,
                    c.seleccion.adverso,
                    _json(dict(c.seleccion.medianas_r)),
                    _json(list(c.seleccion.skus_comparables)),
                    _json(dict(c.seleccion.excluidos)),
                    _json(dict(c.seleccion.familias_excluidas)),
                    _json(dict(c.conteos)),
                    _json(dict(c.porcentajes)),
                    c.mediana_r,
                    c.n_adversos,
                )
                for c in resultado.categorias
            ],
        )
        conn.executemany(
            "INSERT INTO m3_veredictos (run_id, sku, sku_class, veredicto,"
            " familia_campeona, candidato_campeon, n_ventanas, cobertura, razon_sn,"
            " pierde_frente_a_linea_base, comparacion_incompleta, iqr_diferencia_mae,"
            " dm_n_ventanas, dm_horizonte, dm_estadistico, dm_p_valor, dm_p_ajustado,"
            " dm_alfa, dm_significativa, dm_causa, justificacion, modelos_evaluados)"
            f" VALUES ({', '.join('?' * 22)})",
            [
                (
                    run_id,
                    v.sku,
                    v.sku_class,
                    v.veredicto,
                    v.familia_campeona,
                    v.candidato_campeon,
                    v.n_ventanas,
                    v.cobertura,
                    v.razon_sn,
                    v.pierde_frente_a_linea_base,
                    v.comparacion_incompleta,
                    v.iqr_diferencia_mae,
                    v.diebold_mariano.n_ventanas,
                    v.diebold_mariano.horizonte,
                    v.diebold_mariano.estadistico,
                    v.diebold_mariano.p_valor,
                    v.diebold_mariano.p_ajustado,
                    v.diebold_mariano.alfa,
                    v.diebold_mariano.significativa,
                    v.diebold_mariano.causa,
                    v.justificacion,
                    _json([asdict(m) for m in v.modelos_evaluados]),
                )
                for v in resultado.skus
            ],
        )
        _cambiar_estado(conn, run_id, "SELECCIONADA")


def publicar(
    conn: sqlite3.Connection, run_id: str, directorio: str | Path
) -> dict[str, Path]:
    """SELECCIONADA -> PUBLICADA, con el Parquet derivado de SQLite (ADR-03-009).

    Exige veredicto para todo SKU del manifiesto. Las unidades sin marcador no
    pueden existir: cada pronostico referencia la suya. El estado cambia antes
    de exportar, dentro de la transaccion, para que el Parquet de m3_corridas
    diga PUBLICADA; si la exportacion falla, se revierte. Repetir reexporta
    (`.tmp` y `os.replace`) y publica; una corrida PUBLICADA no se reescribe.
    """
    destino = Path(directorio) / run_id
    rutas = {tabla: destino / f"{tabla}.parquet" for tabla in TABLAS}
    with _transaccion(conn):
        estado = _exigir_estado(conn, run_id, ("SELECCIONADA", "PUBLICADA"), "publicar")
        if estado == "PUBLICADA":
            return rutas
        sin_veredicto = [
            sku
            for (sku,) in conn.execute(
                "SELECT sku FROM m3_skus WHERE run_id = ?1"
                " EXCEPT SELECT sku FROM m3_veredictos WHERE run_id = ?1"
                " ORDER BY sku",
                (run_id,),
            )
        ]
        if sin_veredicto:
            raise _inconsistente(
                f"corrida {run_id}: {len(sin_veredicto)} SKU sin veredicto, "
                f"por ejemplo {sin_veredicto[0]}"
            )
        _cambiar_estado(conn, run_id, "PUBLICADA")
        destino.mkdir(parents=True, exist_ok=True)
        evidencia = consultar(conn, run_id)
        for tabla, ruta in rutas.items():
            _escribir_parquet(evidencia[tabla], ruta)
    return rutas


def estado_corrida(conn: sqlite3.Connection, run_id: str) -> EstadoCorrida | None:
    """Estado actual, o `None` si la corrida no esta registrada."""
    fila = conn.execute(
        "SELECT estado FROM m3_corridas WHERE run_id = ?", (run_id,)
    ).fetchone()
    return None if fila is None else fila[0]


def consultar(
    conn: sqlite3.Connection, run_id: str, *, sku: str | None = None
) -> dict[str, pd.DataFrame]:
    """Evidencia de una corrida por tabla, o de un SKU (solo tablas con `sku`)."""
    if sku is None:
        return {tabla: _leer(conn, tabla, "run_id = ?", [run_id]) for tabla in TABLAS}
    return {
        tabla: _leer(conn, tabla, "run_id = ? AND sku = ?", [run_id, sku])
        for tabla in TABLAS_POR_SKU
    }


@contextmanager
def _transaccion(conn: sqlite3.Connection) -> Iterator[None]:
    # SQLite trae las claves foraneas apagadas (ADR-03-009). BEGIN IMMEDIATE toma
    # el bloqueo de escritura antes de leer el estado de la corrida.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException as exc:
        conn.rollback()
        if not isinstance(exc, PersistenciaEvidenciaError):
            _logger.error("Escritura revertida: %s", exc)
        raise
    conn.commit()


def _exigir_estado(
    conn: sqlite3.Connection,
    run_id: str,
    permitidos: Collection[EstadoCorrida],
    operacion: str,
) -> EstadoCorrida:
    estado = estado_corrida(conn, run_id)
    if estado is None or estado not in permitidos:
        _logger.error(
            "Operacion rechazada operacion=%s run_id=%s estado=%s",
            operacion,
            run_id,
            estado,
        )
        raise EstadoCorridaError(run_id, estado, operacion)
    return estado


def _cambiar_estado(conn: sqlite3.Connection, run_id: str, destino: str) -> None:
    conn.execute(
        "UPDATE m3_corridas SET estado = ? WHERE run_id = ?", (destino, run_id)
    )
    _logger.info("Corrida run_id=%s pasa a %s", run_id, destino)


def _inconsistente(mensaje: str) -> EvidenciaInconsistenteError:
    _logger.error("Evidencia inconsistente: %s", mensaje)
    return EvidenciaInconsistenteError(mensaje)


def _filas_metricas(
    run_id: str, evaluacion: EvaluacionSku
) -> Iterator[tuple[Any, ...]]:
    for candidato in (evaluacion.linea_base, *evaluacion.candidatos):
        clave = (run_id, evaluacion.sku, candidato.candidato_id)
        for ventana in candidato.por_ventana:
            yield from _filas(clave, _fecha(ventana.origen), ventana.metricas)
        if candidato.agregadas is not None:
            yield from _filas(clave, VENTANA_AGREGADA, candidato.agregadas)
        for nombre in _CONTEOS:
            yield (*clave, VENTANA_AGREGADA, nombre, getattr(candidato, nombre), None)


def _filas(
    clave: tuple[str, str, str], ventana: str, metricas: Metricas
) -> Iterable[tuple[Any, ...]]:
    for nombre in _METRICAS:
        valor = getattr(metricas, nombre)
        causa = metricas.no_calculables.get(nombre)
        yield (*clave, ventana, nombre, None if valor is None else float(valor), causa)


def _valores(pronostico: PronosticoFechado) -> list[float | None]:
    valores = np.asarray(pronostico.valores, dtype=float)
    if valores.shape != (len(pronostico.fechas),):
        # Forma invalida: 3.4 la cuenta como ventana invalida; se guarda sin valor.
        return [None] * len(pronostico.fechas)
    return [float(v) if np.isfinite(v) else None for v in valores]


def _fecha(instante: pd.Timestamp) -> str:
    return instante.strftime("%Y-%m-%d")


def _json(valor: Any) -> str:
    return json.dumps(valor, sort_keys=True, ensure_ascii=False)


def _leer(
    conn: sqlite3.Connection, tabla: str, filtro: str, parametros: list[str]
) -> pd.DataFrame:
    frame = pd.read_sql_query(
        f"SELECT * FROM {tabla} WHERE {filtro} ORDER BY rowid",
        conn,
        params=parametros,
        dtype_backend="numpy_nullable",
    )
    tipos = {
        columna: _DTYPES[declarado]
        for _, columna, declarado, *_ in conn.execute(f"PRAGMA table_info({tabla})")
    }
    return frame.astype(tipos)


def _escribir_parquet(frame: pd.DataFrame, ruta: Path) -> None:
    temporal = ruta.with_name(ruta.name + ".tmp")
    frame.to_parquet(temporal, index=False)
    os.replace(temporal, ruta)
