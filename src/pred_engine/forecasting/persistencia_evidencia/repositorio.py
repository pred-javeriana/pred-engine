"""Escritura idempotente, publicacion y consulta de la evidencia de M3 (3.5-A1).

ADR-03-009: la evidencia vive en las tablas `m3_*` de la base de la plataforma
(`esquema_m3.sql`, su migracion 0005) y el Parquet solo se deriva al publicar.
ADR-03-010: claves naturales, una unidad confirmada no se reescribe y cada
transicion de estado va en la misma transaccion que el trabajo que la completa.
La conexion la abre quien orquesta; aqui no se crean tablas.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pred_engine.comun.exclusiones import SkuExclusion
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
from pred_engine.forecasting.generador_pronosticos import FalloPronostico
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
    "m3_observaciones",
)
TABLAS = (
    "m3_corridas",
    "m3_protocolos",
    "m3_categorias",
    "m3_cartera",
    "m3_cortes",
    "m3_lanzamientos_tardios",
    *TABLAS_POR_SKU,
)
# Unica causa de M1 que M3 publica como lanzamiento tardio (contrato v2, seccion 5).
CAUSA_LANZAMIENTO_TARDIO = "lanzado despues del corte"

_METRICAS = ("n_pares", "mae", "rmse", "me", "mase", "razon_sn")
_CONTEOS = ("n_ventanas_totales", "n_ventanas_validas", "n_pronosticos_validos")
# Tipo declarado en el DDL -> dtype: una columna toda NULL conserva su tipo.
_DTYPES = {"INTEGER": "Int64", "REAL": "Float64", "TEXT": "string"}
_FILAS_POR_BLOQUE_EXPORTACION = 10_000


def esquema_m3() -> str:
    """DDL inicial de las tablas `m3_*`, publicado como migracion 0005."""
    paquete = files("pred_engine.forecasting.persistencia_evidencia")
    return paquete.joinpath("esquema_m3.sql").read_text(encoding="utf-8")


def registrar_corrida(
    conn: sqlite3.Connection,
    identidad: IdentidadCorrida,
    handoff: HandoffValidado | LoteRechazadoError,
    *,
    ingesta_sha256: str,
    datos_sinteticos: bool | None,
    run_id_m2: str | None = None,
    skus_fallidos: tuple[str, ...] = (),
) -> str:
    """Registra la corrida con el resultado de 3.2 y devuelve su `run_id`.

    Un lote rechazado queda en RECHAZADA con su causa; uno valido, en EN_CURSO
    con los SKUs de su manifiesto, sus fallos de candidato y los fallidos en L2.
    Registrar otra vez
    la misma identidad no cambia nada. `ingesta_sha256` es la clave de
    `ingestas` (archivo fuente de M1); la identidad lleva `ingesta_ref_m1`, que
    es el Parquet publicado.
    """
    if isinstance(handoff, LoteRechazadoError):
        estado, causa, run_id_m2, fallos, skus = (
            "RECHAZADA",
            f"{handoff.motivo}: {handoff.detalle}",
            run_id_m2,
            (),
            set(),
        )
    else:
        if run_id_m2 is not None and run_id_m2 != handoff.run_id_m2:
            raise _inconsistente("referencia M2 incompatible con el handoff")
        if (
            handoff.contexto.ingesta_ref_m1 != identidad.ingesta_ref_m1
            or handoff.contexto.t_corte_reserva != identidad.t_corte_reserva
        ):
            raise _inconsistente("identidad incompatible con la particion del handoff")
        estado, causa, run_id_m2, fallos = (
            "EN_CURSO",
            None,
            handoff.run_id_m2,
            handoff.fallos,
        )
        skus = (
            {c.sku for c in handoff.candidatos}
            | {f.sku for f in fallos}
            | set(skus_fallidos)
        )
    run_id = identidad.run_id
    with _transaccion(conn):
        origenes = conn.execute(
            "SELECT DISTINCT datos_sinteticos FROM m3_corridas "
            "WHERE ingesta_sha256 = ?1 AND origen_declarado=1 UNION "
            "SELECT datos_sinteticos FROM m3_ingestas "
            "WHERE ingesta_sha256 = ?1 AND origen_declarado=1 UNION "
            # Los mismos bytes publicados por otra ingesta tampoco cambian de origen.
            "SELECT datos_sinteticos FROM m3_ingestas "
            "WHERE ingesta_ref_m1 = ?2 AND origen_declarado=1",
            (ingesta_sha256, identidad.ingesta_ref_m1),
        ).fetchall()
        if any(bool(f[0]) != datos_sinteticos for f in origenes):
            raise _inconsistente(
                "el origen real o sintetico de una ingesta no puede cambiar"
            )
        conn.execute(
            "INSERT INTO m3_ingestas "
            "(ingesta_sha256, ingesta_ref_m1, datos_sinteticos, origen_declarado) "
            "VALUES (?, ?, ?, ?) ON CONFLICT (ingesta_sha256, ingesta_ref_m1) "
            "DO UPDATE SET datos_sinteticos=excluded.datos_sinteticos, "
            "origen_declarado=excluded.origen_declarado "
            "WHERE m3_ingestas.origen_declarado=0",
            (
                ingesta_sha256,
                identidad.ingesta_ref_m1,
                bool(datos_sinteticos),
                datos_sinteticos is not None,
            ),
        )
        nueva = conn.execute(
            "INSERT INTO m3_corridas (run_id, ingesta_sha256, run_id_m2, identidad,"
            " datos_sinteticos, estado, causa_rechazo, origen_declarado) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (run_id) DO NOTHING",
            (
                run_id,
                ingesta_sha256,
                run_id_m2,
                identidad.canonica(),
                bool(datos_sinteticos),
                estado,
                causa,
                datos_sinteticos is not None,
            ),
        ).rowcount
        if not nueva:
            previa = conn.execute(
                "SELECT identidad, ingesta_sha256, run_id_m2, datos_sinteticos, "
                "origen_declarado "
                "FROM m3_corridas WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if previa != (
                identidad.canonica(),
                ingesta_sha256,
                run_id_m2,
                bool(datos_sinteticos),
                datos_sinteticos is not None,
            ):
                raise _inconsistente(
                    "referencias u origen de datos incompatibles con la corrida previa"
                )
        if not nueva and isinstance(handoff, LoteRechazadoError):
            conn.execute(
                "UPDATE m3_corridas SET estado='RECHAZADA', causa_rechazo=? "
                "WHERE run_id=? AND estado!='PUBLICADA'",
                (causa, run_id),
            )
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
    conn: sqlite3.Connection,
    run_id: str,
    sku: str,
    serie: SerieCandidato,
    *,
    codigo_version: str | None = None,
    semilla: int | None = None,
    emitido_en: datetime | None = None,
    fallos: Mapping[pd.Timestamp, FalloPronostico] | None = None,
) -> int:
    """Confirma las ventanas de un candidato y devuelve cuantas eran nuevas.

    Cada origen es una unidad (SKU x candidato x ventana). Su marcador y sus
    pronosticos van en la misma transaccion; una unidad ya confirmada no se
    toca aunque llegue con otros valores.
    """
    with _transaccion(conn):
        _exigir_estado(conn, run_id, ("EN_CURSO",), "guardar_unidades")
        nuevas = _insertar_unidades(
            conn,
            run_id,
            sku,
            serie,
            codigo_version=codigo_version,
            semilla=semilla,
            emitido_en=emitido_en,
            fallos=fallos,
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


def _insertar_unidades(
    conn: sqlite3.Connection,
    run_id: str,
    sku: str,
    serie: SerieCandidato,
    *,
    codigo_version: str | None = None,
    semilla: int | None = None,
    emitido_en: datetime | None = None,
    fallos: Mapping[pd.Timestamp, FalloPronostico] | None = None,
) -> int:
    """Escritura de ``guardar_unidades`` dentro de una transaccion del llamador.

    El llamador abre la transaccion y exige la corrida EN_CURSO.
    """
    nuevas = 0
    for pronostico in serie.pronosticos:
        origen = _fecha(pronostico.origen)
        if conn.execute(
            "SELECT 1 FROM m3_unidades WHERE run_id = ? AND sku = ? "
            "AND candidato_id = ? AND origen = ?",
            (run_id, sku, serie.candidato_id, origen),
        ).fetchone():
            continue
        if not len(pronostico.fechas) or not pronostico.fechas.equals(
            pd.date_range(
                pronostico.origen + pd.Timedelta(days=1),
                periods=len(pronostico.fechas),
            )
        ):
            raise _inconsistente("ventana sin horizonte diario completo")
        valores = _valores(pronostico)
        fallo = (fallos or {}).get(pronostico.origen)
        fallo_explicito = fallo is not None
        if fallo is None and any(v is None for v in valores):
            fallo = FalloPronostico(
                sku,
                serie.candidato_id,
                serie.modelo,
                pronostico.origen,
                "valores",
                "pronostico_invalido",
                f"sku={sku} candidato_id={serie.candidato_id}: salida invalida",
            )
        if fallo is not None:
            if (
                fallo.sku != sku
                or fallo.candidato_id != serie.candidato_id
                or fallo.origen != pronostico.origen
                or fallo.modelo != serie.modelo
            ):
                raise _inconsistente("fallo incompatible con su unidad")
            if fallo_explicito:
                valores = [None] * len(pronostico.fechas)
        causa = (
            None
            if fallo is None
            else _json({**asdict(fallo), "origen": _fecha(fallo.origen)})
        )
        confirmada = conn.execute(
            "INSERT INTO m3_unidades (run_id, sku, candidato_id, origen, familia,"
            " modelo, estado, causa, emitido_en, codigo_version, semilla)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            (
                run_id,
                sku,
                serie.candidato_id,
                origen,
                serie.familia,
                serie.modelo,
                "confirmada" if fallo is None else "fallida",
                causa,
                None if emitido_en is None else emitido_en.isoformat(),
                codigo_version,
                semilla,
            ),
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
                    zip(pronostico.fechas, valores, strict=True),
                    start=1,
                )
            ],
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
            protocolo = conn.execute(
                "SELECT n_unidades_previstas FROM m3_protocolos WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if protocolo is not None:
                confirmadas = conn.execute(
                    "SELECT COUNT(*) FROM m3_unidades WHERE run_id = ?", (run_id,)
                ).fetchone()[0]
                if confirmadas != protocolo[0]:
                    raise _inconsistente(
                        f"corrida {run_id}: {confirmadas} unidades "
                        f"de {protocolo[0]} previstas"
                    )
            _cambiar_estado(conn, run_id, "EVALUADA")


def guardar_evaluacion(
    conn: sqlite3.Connection,
    run_id: str,
    evaluaciones: Sequence[EvaluacionSku],
    resultado: ResultadoEvaluacion,
    *,
    observaciones: Mapping[str, pd.Series],
) -> None:
    """Metricas, seleccion, veredictos y demanda observada: EVALUADA -> SELECCIONADA.

    Todo en una transaccion; en una corrida ya SELECCIONADA no escribe nada.
    Las versiones de politica y metricas deben ser las de la identidad, el
    origen de los datos el de la corrida, y cada modelo evaluado con ventanas,
    incluida la linea base, debe tener sus pronosticos confirmados.
    ``observaciones`` es la reserva real de cada SKU del manifiesto, la misma
    que puntuo las metricas; M4 la muestra frente a los pronosticos, asi que
    cada fecha pronosticada necesita su fila (``valor`` NULL sin observacion).
    """
    with _transaccion(conn):
        estado = _exigir_estado(
            conn, run_id, ("EVALUADA", "SELECCIONADA"), "guardar_evaluacion"
        )
        if estado == "SELECCIONADA":
            _logger.info("Evaluacion ya guardada run_id=%s; sin cambios", run_id)
            return
        texto, datos_sinteticos, origen_declarado = conn.execute(
            "SELECT identidad, datos_sinteticos, origen_declarado "
            "FROM m3_corridas WHERE run_id = ?",
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
        if resultado.datos_sinteticos != (
            bool(datos_sinteticos) if origen_declarado else None
        ):
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
        for e in evaluaciones:
            for c in (e.linea_base, *e.candidatos):
                origenes = {
                    pd.Timestamp(f[0])
                    for f in conn.execute(
                        "SELECT origen FROM m3_unidades WHERE run_id = ? "
                        "AND sku = ? AND candidato_id = ?",
                        (run_id, e.sku, c.candidato_id),
                    )
                }
                if len(origenes) != c.n_ventanas_totales or any(
                    v.origen not in origenes for v in c.por_ventana
                ):
                    raise _inconsistente(
                        f"corrida {run_id}: ventanas sin confirmar "
                        f"para {e.sku}/{c.candidato_id}"
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
        conn.executemany(
            "INSERT INTO m3_cartera (run_id, sku_class, familia, metrica, n_skus, "
            "n_instancias, n_calculables, n_no_calculables, media, mediana, p25, p75) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _filas_cartera(run_id, evaluaciones),
        )
        _insertar_observaciones(conn, run_id, observaciones)
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
        destino.parent.mkdir(parents=True, exist_ok=True)
        # Ningun archivo final aparece hasta que termina la exportacion completa.
        with tempfile.TemporaryDirectory(
            prefix=f".{run_id}-", dir=destino.parent
        ) as temporal:
            preparado = Path(temporal)
            for tabla, ruta in rutas.items():
                _exportar_tabla(conn, run_id, tabla, preparado / ruta.name)
            if destino.exists():
                # Puede quedar una exportacion completa de una caida anterior al
                # commit. Solo se reemplaza el directorio propio de esta corrida.
                esperados = {
                    p.name
                    for ruta in rutas.values()
                    for p in (ruta, ruta.with_suffix(".csv"))
                }
                if any(p.name not in esperados for p in destino.iterdir()):
                    raise _inconsistente(
                        "el destino de publicacion contiene archivos ajenos"
                    )
                shutil.rmtree(destino)
            os.replace(preparado, destino)
    return rutas


def veredictos_publicados(
    conn: sqlite3.Connection, run_id: str
) -> tuple[dict[str, Any], ...]:
    """Consolida evidencia publicada por SKU, sin recalcular sus metricas."""
    cursor = conn.execute(
        "SELECT * FROM evidencia_publicada WHERE run_id = ? ORDER BY sku", (run_id,)
    )
    columnas = [c[0] for c in cursor.description]
    resultado = []
    for fila in cursor.fetchall():
        datos = dict(zip(columnas, fila, strict=True))
        sku = datos["sku"]
        metricas = []
        for cid in (datos["candidato_campeon"], f"{sku}/seasonal_naive/seasonal_naive"):
            metricas.append(
                {
                    nombre: {"valor": valor, "causa": causa}
                    for nombre, valor, causa in conn.execute(
                        "SELECT metrica, valor, causa FROM m3_metricas "
                        "WHERE run_id = ? AND sku = ? AND candidato_id = ? "
                        "AND ventana = ? ORDER BY rowid",
                        (run_id, sku, cid, VENTANA_AGREGADA),
                    )
                }
            )
        datos["modelos_evaluados"] = json.loads(datos["modelos_evaluados"])
        resultado.append(
            {
                "metricas_campeon": metricas[0],
                "metricas_linea_base": metricas[1],
                **datos,
            }
        )
    return tuple(resultado)


def contar_fallos(conn: sqlite3.Connection, run_id: str) -> int:
    """Fallos locales de adaptacion y de ventanas, sin resultados adversos."""
    return conn.execute(
        "SELECT (SELECT COUNT(*) FROM m3_fallos WHERE run_id = ?1) + "
        "(SELECT COUNT(*) FROM m3_unidades WHERE run_id = ?1 AND estado = 'fallida')",
        (run_id,),
    ).fetchone()[0]


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
        conn.commit()
    except BaseException as exc:
        conn.rollback()
        if not isinstance(exc, PersistenciaEvidenciaError):
            _logger.error("Escritura revertida: %s", exc)
        raise


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


def _fijar_corte(
    conn: sqlite3.Connection,
    run_id: str,
    estado: EstadoCorrida,
    t_star: pd.Timestamp,
    lanzamientos_tardios: Sequence[SkuExclusion],
) -> None:
    """Corte y lanzamientos tardios de M1, dentro de la transaccion del llamador.

    Se escriben una sola vez y despues solo se comparan. Una corrida que ya paso
    de EVALUADA sin ellos (anterior a la version 10) no los recibe: sus
    observaciones tampoco existen y M4 responde ``metadata_missing``.
    """
    tardias = sorted({e.sku_id for e in lanzamientos_tardios})
    otras = sorted({e.cause for e in lanzamientos_tardios} - {CAUSA_LANZAMIENTO_TARDIO})
    if otras or len(tardias) != len(lanzamientos_tardios):
        raise _inconsistente(
            f"corrida {run_id}: lanzamientos tardios duplicados o con otra causa"
        )
    previo = conn.execute(
        "SELECT t_star FROM m3_cortes WHERE run_id = ?", (run_id,)
    ).fetchone()
    if previo is None:
        if estado not in ("EN_CURSO", "EVALUADA"):
            return
        conn.execute(
            "INSERT INTO m3_cortes (run_id, t_star) VALUES (?, ?)",
            (run_id, _fecha(t_star)),
        )
        conn.executemany(
            "INSERT INTO m3_lanzamientos_tardios (run_id, sku, causa) VALUES (?, ?, ?)",
            [(run_id, sku, CAUSA_LANZAMIENTO_TARDIO) for sku in tardias],
        )
        return
    guardadas = [
        sku
        for (sku,) in conn.execute(
            "SELECT sku FROM m3_lanzamientos_tardios WHERE run_id = ? ORDER BY sku",
            (run_id,),
        )
    ]
    if (previo[0], guardadas) != (_fecha(t_star), tardias):
        raise _inconsistente(
            f"corrida {run_id}: corte o lanzamientos tardios distintos de los fijados"
        )


def _insertar_observaciones(
    conn: sqlite3.Connection, run_id: str, observaciones: Mapping[str, pd.Series]
) -> None:
    skus = {
        s
        for (s,) in conn.execute("SELECT sku FROM m3_skus WHERE run_id = ?", (run_id,))
    }
    if set(observaciones) != skus:
        raise _inconsistente(
            f"corrida {run_id}: la demanda observada no cubre exactamente los SKU "
            "del manifiesto"
        )
    if (
        conn.execute("SELECT 1 FROM m3_cortes WHERE run_id = ?", (run_id,)).fetchone()
        is None
    ):
        raise _inconsistente(f"corrida {run_id}: sin corte fijado para la reserva")
    filas = []
    for sku in sorted(observaciones):
        serie = observaciones[sku]
        if not isinstance(serie.index, pd.DatetimeIndex) or serie.index.has_duplicates:
            raise _inconsistente(f"corrida {run_id}: fechas invalidas en {sku}")
        valores = serie.to_numpy(dtype=float, na_value=np.nan)
        filas.extend(
            (run_id, sku, _fecha(fecha), float(v) if np.isfinite(v) else None)
            for fecha, v in zip(serie.index, valores, strict=True)
        )
    try:
        conn.executemany(
            "INSERT INTO m3_observaciones (run_id, sku, fecha, valor) "
            "VALUES (?, ?, ?, ?)",
            filas,
        )
    except sqlite3.IntegrityError as exc:
        raise _inconsistente(
            f"corrida {run_id}: demanda observada fuera de la reserva: {exc}"
        ) from exc
    sin_observacion = conn.execute(
        "SELECT sku, fecha FROM m3_pronosticos WHERE run_id = ?1 EXCEPT "
        "SELECT sku, fecha FROM m3_observaciones WHERE run_id = ?1 "
        "ORDER BY sku, fecha LIMIT 1",
        (run_id,),
    ).fetchone()
    if sin_observacion is not None:
        raise _inconsistente(
            f"corrida {run_id}: fecha pronosticada sin fila de demanda observada, "
            f"por ejemplo {sin_observacion}"
        )


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


def _filas_cartera(
    run_id: str, evaluaciones: Sequence[EvaluacionSku]
) -> Iterable[tuple[Any, ...]]:
    grupos: dict[tuple[str, str, str], list[tuple[str, float | None]]] = {}
    for e in evaluaciones:
        for c in (e.linea_base, *e.candidatos):
            for metrica in ("mae", "rmse", "mase", "me", "razon_sn"):
                valor = None if c.agregadas is None else getattr(c.agregadas, metrica)
                grupos.setdefault((e.sku_class, c.familia, metrica), []).append(
                    (e.sku, valor)
                )
    for (clase, familia, metrica), instancias in sorted(grupos.items()):
        valores = [v for _, v in instancias if v is not None and np.isfinite(v)]
        resumen: tuple[float | None, ...] = (None,) * 4
        if valores:
            resumen = tuple(
                float(v)
                for v in (
                    np.mean(valores),
                    np.median(valores),
                    np.percentile(valores, 25),
                    np.percentile(valores, 75),
                )
            )
        yield (
            run_id,
            clase,
            familia,
            metrica,
            len({s for s, _ in instancias}),
            len(instancias),
            len(valores),
            len(instancias) - len(valores),
            *resumen,
        )


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
    return frame.astype(_tipos_sqlite(conn, tabla))


def _tipos_sqlite(conn: sqlite3.Connection, tabla: str) -> dict[str, str]:
    return {
        columna: _DTYPES[declarado]
        for _, columna, declarado, *_ in conn.execute(f"PRAGMA table_info({tabla})")
    }


def _exportar_tabla(
    conn: sqlite3.Connection, run_id: str, tabla: str, ruta: Path
) -> None:
    tipos = _tipos_sqlite(conn, tabla)
    # El esquema no depende del primer bloque: puede estar vacio o todo NULL.
    vacio = pd.DataFrame({columna: pd.Series(dtype=t) for columna, t in tipos.items()})
    esquema = pa.Table.from_pandas(vacio, preserve_index=False).schema
    bloques = pd.read_sql_query(
        f"SELECT * FROM {tabla} WHERE run_id = ? ORDER BY rowid",
        conn,
        params=[run_id],
        dtype_backend="numpy_nullable",
        chunksize=_FILAS_POR_BLOQUE_EXPORTACION,
    )
    with (
        pq.ParquetWriter(ruta, esquema, compression="snappy") as escritor,
        ruta.with_suffix(".csv").open("w", encoding="utf-8", newline="") as csv,
    ):
        primero = True
        for bloque in bloques:
            frame = bloque.astype(tipos)
            escritor.write_table(pa.Table.from_pandas(frame, preserve_index=False))
            frame.to_csv(csv, index=False, header=primero)
            primero = False
