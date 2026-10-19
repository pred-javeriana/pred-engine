"""Calendario diario de la reserva y compuertas anteriores al ajuste (3.1)."""

from __future__ import annotations

from typing import NoReturn

import numpy as np
import pandas as pd

from pred_engine.comun.logger import get_logger
from pred_engine.comun.reserva import RESERVE_FRACTION, ReserveCut
from pred_engine.forecasting.generador_pronosticos.contratos import (
    CodigoExclusion,
    ConfiguracionPronosticos,
    SerieNoElegibleError,
    VentanaReserva,
)

_logger = get_logger(__name__)


def ventanas_reserva(corte: ReserveCut, horizonte: int) -> tuple[VentanaReserva, ...]:
    """Todos los origenes diarios desde t* cuyo horizonte completo esta reservado.

    El ultimo origen es ``last_observed - H dias``. No depende del paso que
    uso el HPO y no genera una ventana parcial al final.
    """
    if type(horizonte) is not int or horizonte < 1:
        raise ValueError("horizonte debe ser un entero positivo")
    if (
        corte.fraction != RESERVE_FRACTION
        or corte.reserved_days < 1
        or corte.first_reserved != corte.t_star + pd.Timedelta(days=1)
        or corte.last_observed != corte.t_star + pd.Timedelta(days=corte.reserved_days)
        or corte.t_star != corte.t_star.normalize()
    ):
        raise ValueError(
            "corte de reserva inconsistente con el protocolo diario del 20 %"
        )
    return tuple(
        VentanaReserva(
            origen=origen,
            fechas=pd.date_range(origen + pd.Timedelta(days=1), periods=horizonte),
        )
        for origen in pd.date_range(
            corte.t_star, periods=max(0, corte.reserved_days - horizonte + 1)
        )
    )


def ventanas_observables(
    corte: ReserveCut, horizonte: int, fechas: pd.Index
) -> tuple[VentanaReserva, ...]:
    """Conserva H completo y omite origenes sin contexto diario observado."""
    ausentes = np.cumsum(~corte.reserved_dates.isin(fechas))
    return tuple(
        v
        for v in ventanas_reserva(corte, horizonte)
        if v.origen == corte.t_star
        or ausentes[(v.origen - corte.first_reserved).days] == 0
    )


def separar_serie(
    sku: str,
    serie: pd.Series,
    corte: ReserveCut,
    configuracion: ConfiguracionPronosticos,
) -> tuple[np.ndarray, pd.Series]:
    """Valida historia diaria y devuelve historia <= t* y reserva sin imputar.

    Se ignoran filas posteriores al fin declarado de la reserva. Una fecha
    ausente en la reserva se conserva como NaN: no se inventa demanda cero.
    """
    if not isinstance(serie.index, pd.DatetimeIndex):
        _excluir(sku, "CALENDAR_GAPS", "la serie necesita un indice de fechas diario")
    indice = serie.index
    if (
        indice.tz is not None
        or indice.hasnans
        or indice.has_duplicates
        or not indice.equals(pd.DatetimeIndex(pd.Series(indice).dt.normalize()))
    ):
        _excluir(sku, "CALENDAR_GAPS", "fechas repetidas, invalidas o no diarias")
    ordenada = serie.sort_index()
    historia = ordenada.loc[ordenada.index <= corte.t_star]
    if len(historia) < configuracion.min_historia:
        _excluir(
            sku,
            "INSUFFICIENT_HISTORY",
            f"{len(historia)} dias, se requieren {configuracion.min_historia}",
        )
    esperado = pd.date_range(historia.index[0], corte.t_star)
    if not historia.index.equals(esperado):
        _excluir(sku, "CALENDAR_GAPS", "faltan dias en la historia hasta t*")
    try:
        valores = historia.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError):
        _excluir(sku, "INVALID_HISTORY", "la historia no contiene demanda numerica")
    if not np.isfinite(valores).all() or (valores < 0).any():
        _excluir(sku, "INVALID_HISTORY", "la historia tiene demanda invalida")
    if configuracion.horizonte > corte.reserved_days:
        _excluir(
            sku, "INSUFFICIENT_RESERVE", "la reserva no contiene un horizonte completo"
        )
    return valores, ordenada.reindex(corte.reserved_dates)


def _excluir(sku: str, codigo: CodigoExclusion, detalle: str) -> NoReturn:
    _logger.error("Serie excluida sku=%s codigo=%s: %s", sku, codigo, detalle)
    raise SerieNoElegibleError(sku, codigo, detalle)
