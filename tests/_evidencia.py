"""Base SQLite y evidencia de M3 para las pruebas de 3.5.

No empieza con `test_`: pytest no lo recolecta; lo importan los `test_*.py`.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from pred_engine.comun.modelos.manifiesto_candidatos import Candidato, ContextoParticion
from pred_engine.forecasting.adaptador_candidatos import (
    FalloCandidato,
    HandoffValidado,
)
from pred_engine.forecasting.evaluaciones.calculo_errores import (
    VERSION_METRICAS,
    BloquesReserva,
    EntradaSku,
    EvaluacionBloques,
    SerieCandidato,
    evaluar_bloques,
    pronosticar_linea_base,
)
from pred_engine.forecasting.evaluaciones.veredictos import (
    POLITICA_INICIAL,
    PoliticaSeleccion,
    ResultadoEvaluacion,
    emitir_veredictos,
)
from pred_engine.forecasting.persistencia_evidencia import (
    IdentidadCorrida,
    aplicar_migraciones_m3,
    esquema_m3,
    guardar_unidades,
    marcar_evaluada,
    registrar_corrida,
)
from tests._evaluaciones import corte, pronostico, reserva, serie
from tests._manifiestos import candidato

INGESTA = "c" * 64

# Solo la columna que referencia M3; la tabla real es de la plataforma (0001).
_INGESTAS = (
    "CREATE TABLE ingestas (id INTEGER PRIMARY KEY, sha256 TEXT UNIQUE);"
    "CREATE TABLE ejecuciones (id INTEGER PRIMARY KEY);"
)

# S2 es lumpy: ML no es elegible ahi y Chronos no llega en el handoff.
_SKUS = (("S1", "smooth"), ("S2", "lumpy"))
_FAMILIAS = (("classical", "sarima", 9.0), ("ml", "lightgbm", 12.0))
_CANDIDATO: TypeAdapter[Candidato] = TypeAdapter(Candidato)


def base() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(_INGESTAS + esquema_m3())
    conn.execute("PRAGMA user_version = 5")
    aplicar_migraciones_m3(conn)
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
        contexto=ContextoParticion(
            ingesta_ref_m1="b" * 64,
            t_corte_reserva=date(2024, 1, 14),
            fraccion_reserva=0.2,
        ),
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
# Reserva real de 4 dias, constante en 10.
_HISTORIA = np.r_[np.full(7, 13.0), np.full(7, 6.0)]
# La ventana con origen t* es el bloque de seleccion y la de t*+2 el de prueba;
# la politica acepta esa unica ventana por bloque. Misma version que la inicial.
POLITICA = PoliticaSeleccion(n_min=1, fraccion_seleccion=0.5)
BLOQUES = BloquesReserva.de(corte(4), 2, POLITICA.fraccion_seleccion)


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
) -> tuple[str, list[EvaluacionBloques], ResultadoEvaluacion]:
    """Corrida en EVALUADA con sus unidades y la evaluacion real de 3.4.

    Como orquestaria 3.3: las unidades de cada candidato y las de Seasonal
    Naive, que genera 3.4 en las mismas ventanas.
    """
    run_id = corrida(conn, *fallos)
    for entrada in entradas():
        series = list(entrada.candidatos)
        if con_linea_base:
            series.append(pronosticar_linea_base(entrada, corte(4)))
        for serie_candidato in series:
            guardar_unidades(conn, run_id, entrada.sku, serie_candidato)
    marcar_evaluada(conn, run_id)
    # El corte lo fija el protocolo; aqui se siembra para probar solo 3.4 y 3.5.
    conn.execute(
        "INSERT INTO m3_cortes (run_id, t_star) VALUES (?, ?)",
        (run_id, corte(4).t_star.strftime("%Y-%m-%d")),
    )
    conn.commit()
    evaluaciones = [evaluar_bloques(e, corte(4), BLOQUES) for e in entradas()]
    resultado = emitir_veredictos(
        evaluaciones, BLOQUES, datos_sinteticos=False, politica=POLITICA
    )
    return run_id, evaluaciones, resultado


def observaciones(conn: sqlite3.Connection, run_id: str) -> dict[str, pd.Series]:
    """La reserva real de `entradas()` para cada SKU del manifiesto de la corrida."""
    skus = [
        s
        for (s,) in conn.execute("SELECT sku FROM m3_skus WHERE run_id = ?", (run_id,))
    ]
    return {sku: reserva([10.0] * 4) for sku in skus}


def filas(conn: sqlite3.Connection, tabla: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {tabla}").fetchone()[0]
