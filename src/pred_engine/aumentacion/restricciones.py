"""Compuerta de restricciones fisicas de la seccion 0.3.

Impone las leyes de conservacion logistica sobre el panel sintetico producido
por el Moving Block Bootstrap (``pred_engine.aumentacion.mbb``):

* A1 - Demanda no negativa (clipping asimetrico).
* A2 - Discretizacion estricta (truncamiento entero).
* A3 - Limites de lead time derivados de la semilla.

Todas las transformaciones son puras y vectorizadas: reciben ``numpy`` /
``pandas`` y devuelven copias nuevas sin mutar la entrada.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pred_engine.aumentacion.errores import PhysicalConstraintError
from pred_engine.comun.logger import get_logger

_logger = get_logger(__name__)

# Columnas del panel sobre las que opera la compuerta (alineadas con el
# contrato de datos de la seccion 0.4 y con el pipeline de ingesta del Modulo 1).
_COLUMNAS_PANEL: tuple[str, ...] = (
    "sku_id",
    "timestamp",
    "demand_qty",
    "lead_time_days",
)


@dataclass(frozen=True, slots=True)
class ResultadoRectificacion:
    """Salida del clipping asimetrico: copia rectificada + metricas."""

    valores: np.ndarray
    n_rectificadas: int
    porcentaje: float


@dataclass(frozen=True, slots=True)
class LimitesLeadTime:
    """Umbrales de tiempo de entrega derivados de la semilla original."""

    minimo: int
    maximo: int

    def __post_init__(self) -> None:
        if self.minimo < 1:
            raise PhysicalConstraintError(
                f"el limite inferior de lead time debe ser >= 1 (dado {self.minimo})"
            )
        if self.maximo < self.minimo:
            raise PhysicalConstraintError(
                "el limite superior de lead time no puede ser menor que el inferior"
            )


@dataclass(frozen=True, slots=True)
class ResultadoAcotamiento:
    """Salida del acotamiento de lead time: copia acotada + metricas."""

    valores: np.ndarray
    n_acotadas: int
    limites: LimitesLeadTime


def rectificar_demanda_no_negativa(
    demanda: np.ndarray,
    *,
    umbral_inferior: float = 0.0,
) -> ResultadoRectificacion:
    """A1 - Eleva a ``umbral_inferior`` toda demanda por debajo del umbral.

    Clipping asimetrico: solo se toca la cola inferior; los valores superiores
    permanecen intactos. No muta el arreglo de entrada.
    """
    entrada = np.asarray(demanda, dtype="float64")
    if entrada.ndim != 1:
        raise PhysicalConstraintError("demanda debe ser un arreglo 1D")

    rectificada = np.maximum(entrada, float(umbral_inferior))
    mascara = entrada < float(umbral_inferior)
    n_rectificadas = int(np.count_nonzero(mascara))
    total = int(entrada.size)
    porcentaje = (n_rectificadas / total * 100.0) if total else 0.0

    if n_rectificadas:
        _logger.info(
            "Clipping asimetrico: %s/%s observaciones rectificadas (%.2f%%)",
            n_rectificadas,
            total,
            porcentaje,
        )
    return ResultadoRectificacion(
        valores=rectificada,
        n_rectificadas=n_rectificadas,
        porcentaje=porcentaje,
    )


def truncar_a_unidades_enteras(demanda: np.ndarray) -> np.ndarray:
    """A2 - Trunca cada observacion a su valor entero absoluto.

    Se elige truncamiento (``trunc``) y no redondeo (``round``): una unidad
    fisica indivisible solo se contabiliza cuando esta completa, de modo que
    2.99 kits equivalen a 2 kits despachables, nunca a 3.
    """
    entrada = np.asarray(demanda, dtype="float64")
    if entrada.ndim != 1:
        raise PhysicalConstraintError("demanda debe ser un arreglo 1D")
    return np.trunc(np.abs(entrada)).astype("int64")


def limites_lead_time_desde_semilla(semilla_lead_time: np.ndarray) -> LimitesLeadTime:
    """A3 - Deriva los umbrales min/max de lead time de la semilla original.

    El limite inferior nunca baja de 1 dia (un lead time de 0 invalidaria los
    calculos de stock de seguridad posteriores).
    """
    entrada = np.asarray(semilla_lead_time, dtype="float64")
    if entrada.size == 0:
        raise PhysicalConstraintError("la semilla de lead time no puede estar vacia")
    finitos = entrada[np.isfinite(entrada)]
    if finitos.size == 0:
        raise PhysicalConstraintError(
            "la semilla de lead time no tiene valores finitos"
        )

    minimo = max(1, int(np.floor(finitos.min())))
    maximo = max(minimo, int(np.ceil(finitos.max())))
    return LimitesLeadTime(minimo=minimo, maximo=maximo)


def acotar_lead_time(
    lead_time: np.ndarray,
    limites: LimitesLeadTime,
) -> ResultadoAcotamiento:
    """A3 - Acota el lead time simulado al rango derivado de la semilla.

    Emite una advertencia trazable por cada corrida en la que se debio acotar
    al menos una observacion.
    """
    entrada = np.asarray(lead_time, dtype="float64")
    if entrada.ndim != 1:
        raise PhysicalConstraintError("lead_time debe ser un arreglo 1D")

    acotada = np.clip(np.trunc(entrada), limites.minimo, limites.maximo)
    n_acotadas = int(np.count_nonzero(acotada != np.trunc(entrada)))
    if n_acotadas:
        _logger.warning(
            "Lead time fuera de rango: %s observaciones acotadas a [%s, %s]",
            n_acotadas,
            limites.minimo,
            limites.maximo,
        )
    return ResultadoAcotamiento(
        valores=acotada.astype("int64"),
        n_acotadas=n_acotadas,
        limites=limites,
    )


@dataclass(frozen=True, slots=True)
class ResultadoCompuerta:
    """Panel rectificado por la compuerta fisica + metricas agregadas."""

    panel: pd.DataFrame
    n_demanda_rectificada: int
    n_lead_time_acotado: int
    limites_lead_time: LimitesLeadTime


def aplicar_restricciones_fisicas(
    panel: pd.DataFrame,
    *,
    limites_lead_time: LimitesLeadTime,
) -> ResultadoCompuerta:
    """Aplica A1 + A2 + A3 sobre el panel sintetico. No muta el DataFrame."""
    faltan = [c for c in _COLUMNAS_PANEL if c not in panel.columns]
    if faltan:
        raise PhysicalConstraintError(f"al panel le faltan columnas: {faltan}")
    if panel.empty:
        raise PhysicalConstraintError("el panel sintetico no tiene filas")

    salida = panel.loc[:, list(_COLUMNAS_PANEL)].copy()

    rectificacion = rectificar_demanda_no_negativa(salida["demand_qty"].to_numpy())
    demanda_entera = truncar_a_unidades_enteras(rectificacion.valores)
    acotamiento = acotar_lead_time(
        salida["lead_time_days"].to_numpy(),
        limites_lead_time,
    )

    salida["demand_qty"] = demanda_entera
    salida["lead_time_days"] = acotamiento.valores
    salida["sku_id"] = salida["sku_id"].astype("string")
    salida["timestamp"] = pd.to_datetime(salida["timestamp"]).dt.normalize()

    if (salida["demand_qty"].to_numpy() < 0).any():
        raise PhysicalConstraintError("quedo demanda negativa tras la compuerta")
    if (salida["lead_time_days"].to_numpy() < 1).any():
        raise PhysicalConstraintError("quedo lead time < 1 tras la compuerta")

    _logger.info(
        "Compuerta fisica superada: %s filas, %s demandas rectificadas, "
        "%s lead times acotados",
        len(salida),
        rectificacion.n_rectificadas,
        acotamiento.n_acotadas,
    )
    return ResultadoCompuerta(
        panel=salida,
        n_demanda_rectificada=rectificacion.n_rectificadas,
        n_lead_time_acotado=acotamiento.n_acotadas,
        limites_lead_time=limites_lead_time,
    )
