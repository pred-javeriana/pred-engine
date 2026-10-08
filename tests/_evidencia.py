"""Base SQLite y evidencia de M3 para las pruebas de 3.5.

No empieza con `test_`: pytest no lo recolecta; lo importan los `test_*.py`.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from typing import Any

import numpy as np
from pydantic import TypeAdapter

from pred_engine.comun.modelos.manifiesto_candidatos import Candidato
from pred_engine.forecasting.adaptador_candidatos import (
    FalloCandidato,
    HandoffValidado,
)
from pred_engine.forecasting.evaluaciones.calculo_errores import (
    VERSION_METRICAS,
    EntradaSku,
    EvaluacionSku,
    SerieCandidato,
    evaluar_sku,
    pronosticar_linea_base,
)
from pred_engine.forecasting.evaluaciones.veredictos import (
    POLITICA_INICIAL,
    ResultadoEvaluacion,
    emitir_veredictos,
)
from pred_engine.forecasting.persistencia_evidencia import (
    IdentidadCorrida,
    esquema_m3,
    guardar_unidades,
    marcar_evaluada,
    registrar_corrida,
)
from tests._evaluaciones import corte, pronostico, reserva, serie
from tests._manifiestos import CONTEXTO, candidato

INGESTA = "c" * 64

# Solo la columna que referencia M3; la tabla real es de la plataforma (0001).
_INGESTAS = "CREATE TABLE ingestas (id INTEGER PRIMARY KEY, sha256 TEXT UNIQUE);"

# S2 es lumpy: ML no es elegible ahi y Chronos no llega en el handoff.
_SKUS = (("S1", "smooth"), ("S2", "lumpy"))
_FAMILIAS = (("classical", "sarima", 9.0), ("ml", "lightgbm", 12.0))
_CANDIDATO: TypeAdapter[Candidato] = TypeAdapter(Candidato)


def base() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(_INGESTAS + esquema_m3())
    conn.execute("INSERT INTO ingestas (sha256) VALUES (?)", (INGESTA,))
    conn.commit()
    return conn


def identidad(**cambios: Any) -> IdentidadCorrida:
    datos: dict[str, Any] = {
        "ingesta_ref_m1": "b" * 64,
        "configuracion_id": "m2-corrida-1",
        "codigo_version": "0123456789abcdef",
        "t_corte_reserva": date(2024, 1, 14),
        "version_politica": POLITICA_INICIAL.version,
        "version_metricas": VERSION_METRICAS,
    }
    return IdentidadCorrida(**(datos | cambios))


def fallo(sku: str = "S1", familia: str = "dl") -> FalloCandidato:
    return FalloCandidato(
        sku=sku,
        candidato_id=f"{sku}/{familia}/mlp",
        modelo="mlp",
        campo="configuracion.lags",
        codigo_error="fuera_de_dominio",
        mensaje=f"sku={sku} modelo=mlp campo=configuracion.lags: fuera de rango",
    )


def handoff(*fallos: FalloCandidato) -> HandoffValidado:
    """Manifiesto validado por 3.2: SARIMA y LightGBM para S1 y S2."""
    return HandoffValidado(
        contexto=CONTEXTO,
        run_id_m2="m2-corrida-1",
        candidatos=tuple(
            _CANDIDATO.validate_python(candidato(familia, sku, sku_class=sku_class))
            for sku, sku_class in _SKUS
            for familia, _, _ in _FAMILIAS
        ),
        fallos=fallos,
    )


def corrida(
    conn: sqlite3.Connection, *fallos: FalloCandidato, datos_sinteticos: bool = False
) -> str:
    return registrar_corrida(
        conn,
        identidad(),
        handoff(*fallos),
        ingesta_sha256=INGESTA,
        datos_sinteticos=datos_sinteticos,
    )


# --- Dos SKUs, dos candidatos y la linea base, con dos ventanas de 2 dias ----

# Q1 = 7 (MASE calculable) y Seasonal Naive pronostica 6 en ambas ventanas.
# Reserva real constante en 10.
_HISTORIA = np.r_[np.full(7, 13.0), np.full(7, 6.0)]


def _serie(sku: str, familia: str, modelo: str, valores: float) -> SerieCandidato:
    return serie(
        f"{sku}/{familia}/{modelo}",
        [pronostico(0, [valores, valores]), pronostico(2, [valores, valores])],
        familia,
        modelo,
    )


def entradas() -> list[EntradaSku]:
    return [
        EntradaSku(
            sku=sku,
            sku_class=sku_class,
            historia=_HISTORIA,
            reserva=reserva([10.0] * 4),
            candidatos=tuple(_serie(sku, f, m, v) for f, m, v in _FAMILIAS),
        )
        for sku, sku_class in _SKUS
    ]


def corrida_evaluada(
    conn: sqlite3.Connection,
    *fallos: FalloCandidato,
    con_linea_base: bool = True,
) -> tuple[str, list[EvaluacionSku], ResultadoEvaluacion]:
    """Corrida en EVALUADA con sus unidades y la evaluacion real de 3.4.

    Como orquestaria 3.3: las unidades de cada candidato y las de Seasonal
    Naive, que genera 3.4 en las mismas ventanas.
    """
    run_id = corrida(conn, *fallos)
    for entrada in entradas():
        series = list(entrada.candidatos)
        if con_linea_base:
            series.append(pronosticar_linea_base(entrada, corte()))
        for serie_candidato in series:
            guardar_unidades(conn, run_id, entrada.sku, serie_candidato)
    marcar_evaluada(conn, run_id)
    evaluaciones = [evaluar_sku(e, corte()) for e in entradas()]
    return run_id, evaluaciones, emitir_veredictos(evaluaciones, datos_sinteticos=False)


def filas(conn: sqlite3.Connection, tabla: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {tabla}").fetchone()[0]
