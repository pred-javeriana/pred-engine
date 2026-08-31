"""0.3-B1 - Calculador de divergencia parametrica (rejection sampling basico).

Mide cuanto se aleja una serie sintetica candidata de la semilla original en
media y varianza globales. El veredicto y los estadisticos calculados se
devuelven juntos para que la decision de aceptacion sea auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pred_engine.comun.logger import get_logger

_logger = get_logger(__name__)

# Tolerancia por defecto: 5 % de divergencia relativa (seccion 0.3-C).
TOLERANCIA_DIVERGENCIA_POR_DEFECTO = 0.05


@dataclass(frozen=True, slots=True)
class VeredictoDivergencia:
    """Resultado del calculo de divergencia parametrica."""

    aceptada: bool
    tolerancia: float
    media_semilla: float
    media_candidata: float
    varianza_semilla: float
    varianza_candidata: float
    divergencia_media: float
    divergencia_varianza: float


def _divergencia_relativa(referencia: float, candidato: float) -> float:
    """|candidato - referencia| / |referencia|, con guarda para referencia ~ 0."""
    if np.isclose(referencia, 0.0):
        return 0.0 if np.isclose(candidato, 0.0) else float("inf")
    return abs(candidato - referencia) / abs(referencia)


def evaluar_divergencia_parametrica(
    semilla: np.ndarray,
    candidata: np.ndarray,
    *,
    tolerancia: float = TOLERANCIA_DIVERGENCIA_POR_DEFECTO,
) -> VeredictoDivergencia:
    """Compara media y varianza de la candidata contra la semilla.

    La candidata se acepta solo si ambas divergencias relativas quedan dentro
    de ``tolerancia`` (parametro configurable, nunca incrustado).
    """
    if tolerancia <= 0:
        raise ValueError("tolerancia debe ser mayor que 0")

    ref = np.asarray(semilla, dtype="float64")
    cand = np.asarray(candidata, dtype="float64")
    if ref.size == 0 or cand.size == 0:
        raise ValueError("semilla y candidata no pueden estar vacias")

    media_semilla = float(np.mean(ref))
    media_candidata = float(np.mean(cand))
    var_semilla = float(np.var(ref))
    var_candidata = float(np.var(cand))

    div_media = _divergencia_relativa(media_semilla, media_candidata)
    div_var = _divergencia_relativa(var_semilla, var_candidata)
    aceptada = div_media <= tolerancia and div_var <= tolerancia

    _logger.debug(
        "Divergencia parametrica: media=%.4f varianza=%.4f tolerancia=%.4f -> %s",
        div_media,
        div_var,
        tolerancia,
        "aceptada" if aceptada else "rechazada",
    )
    return VeredictoDivergencia(
        aceptada=aceptada,
        tolerancia=tolerancia,
        media_semilla=media_semilla,
        media_candidata=media_candidata,
        varianza_semilla=var_semilla,
        varianza_candidata=var_candidata,
        divergencia_media=div_media,
        divergencia_varianza=div_var,
    )
