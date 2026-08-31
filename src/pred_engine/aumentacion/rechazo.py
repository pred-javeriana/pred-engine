"""0.3-B2 - Bucle de rechazo y remuestreo.

Envuelve al Moving Block Bootstrap del companiero
(``pred_engine.aumentacion.mbb``): por cada serie sintetica solicitada genera
candidatas hasta que una supere la compuerta de divergencia parametrica, o
eleva una excepcion controlada al agotar los reintentos.

El proceso es reproducible: dos corridas con la misma semilla aleatoria
producen exactamente el mismo panel sintetico.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from pred_engine.aumentacion.divergencia import (
    TOLERANCIA_DIVERGENCIA_POR_DEFECTO,
    VeredictoDivergencia,
    evaluar_divergencia_parametrica,
)
from pred_engine.aumentacion.errores import DivergenceRejectionExhausted
from pred_engine.aumentacion.mbb import (
    compose_series,
    decompose_series,
    moving_block_bootstrap,
)
from pred_engine.comun.logger import get_logger

_logger = get_logger(__name__)

# Callable que produce una serie sintetica candidata a partir de un generador
# aleatorio. Se inyecta para pruebas; por defecto usa el motor MBB.
GeneradorCandidata = Callable[[np.random.Generator], np.ndarray]


@dataclass(frozen=True, slots=True)
class ResultadoRechazo:
    """Series aceptadas por la compuerta + trazabilidad de la corrida."""

    series: list[np.ndarray]
    semilla_aleatoria: int
    intentos: int
    rechazos: int
    veredictos: list[VeredictoDivergencia] = field(default_factory=list)

    @property
    def tasa_rechazo(self) -> float:
        return self.rechazos / self.intentos if self.intentos else 0.0


def motor_mbb(
    serie_semilla: np.ndarray,
    *,
    period: int,
    block_size: int = 3,
) -> GeneradorCandidata:
    """Construye un generador de candidatas basado en STL + MBB de residuales."""
    trend, seasonal, residual = decompose_series(
        series_array=np.asarray(serie_semilla, dtype="float64"),
        period=period,
    )

    def generar(rng: np.random.Generator) -> np.ndarray:
        residual_boot = moving_block_bootstrap(
            serie_sku=residual,
            block_size=block_size,
            tamaño_esperado=len(residual),
            rng=rng,
        )
        return compose_series(
            trend=trend,
            seasonal=seasonal,
            residual=residual_boot,
        )

    return generar


def generar_series_aceptadas(
    serie_semilla: np.ndarray,
    *,
    period: int,
    n_series: int = 1,
    block_size: int = 3,
    tolerancia: float = TOLERANCIA_DIVERGENCIA_POR_DEFECTO,
    max_reintentos: int = 20,
    semilla_aleatoria: int = 42,
    generador: GeneradorCandidata | None = None,
) -> ResultadoRechazo:
    """Genera ``n_series`` series sinteticas que respetan la paridad estadistica.

    Parameters
    ----------
    serie_semilla:
        Serie 1D de la semilla original (p. ej. demanda de un SKU de Kaggle).
    period:
        Periodo estacional para la descomposicion STL.
    max_reintentos:
        Reintentos permitidos por serie antes de elevar
        :class:`DivergenceRejectionExhausted`.
    semilla_aleatoria:
        Fija el generador ``numpy`` para garantizar reproducibilidad.
    generador:
        Motor de remuestreo inyectable; por defecto STL + MBB de residuales.
    """
    if n_series <= 0:
        raise ValueError("n_series debe ser mayor que 0")
    if max_reintentos <= 0:
        raise ValueError("max_reintentos debe ser mayor que 0")

    semilla_arr = np.asarray(serie_semilla, dtype="float64")
    rng = np.random.default_rng(semilla_aleatoria)
    motor = generador or motor_mbb(
        semilla_arr,
        period=period,
        block_size=block_size,
    )

    series: list[np.ndarray] = []
    veredictos: list[VeredictoDivergencia] = []
    intentos = 0
    rechazos = 0

    for indice in range(n_series):
        for reintento in range(max_reintentos):
            intentos += 1
            candidata = np.asarray(motor(rng), dtype="float64")
            veredicto = evaluar_divergencia_parametrica(
                semilla_arr,
                candidata,
                tolerancia=tolerancia,
            )
            if veredicto.aceptada:
                series.append(candidata)
                veredictos.append(veredicto)
                break
            rechazos += 1
            _logger.info(
                "Serie %s: candidata rechazada en reintento %s "
                "(div_media=%.4f div_var=%.4f)",
                indice,
                reintento,
                veredicto.divergencia_media,
                veredicto.divergencia_varianza,
            )
        else:
            _logger.error(
                "Rejection sampling agotado para la serie %s tras %s reintentos",
                indice,
                max_reintentos,
            )
            raise DivergenceRejectionExhausted(max_reintentos, tolerancia)

    resultado = ResultadoRechazo(
        series=series,
        semilla_aleatoria=semilla_aleatoria,
        intentos=intentos,
        rechazos=rechazos,
        veredictos=veredictos,
    )
    _logger.info(
        "Rejection sampling completado: %s series, tasa de rechazo %.2f%%",
        len(series),
        resultado.tasa_rechazo * 100.0,
    )
    return resultado
