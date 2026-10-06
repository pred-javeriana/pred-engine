"""Metricas de error sobre la reserva (3.4-A1, ADR-03-006).

Solo recibe arreglos y fechas: no puede reajustar ni tocar un modelo.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pred_engine.comun.logger import get_logger
from pred_engine.comun.reserva import ReserveCut
from pred_engine.comun.walkforward.metricas import mae, rmse
from pred_engine.forecasting.evaluaciones.calculo_errores.contratos import (
    EntradaSku,
    EvaluacionSku,
    Metricas,
    MetricasCandidato,
    MetricasVentana,
    SerieCandidato,
)
from pred_engine.forecasting.evaluaciones.calculo_errores.errores import (
    EvaluacionRetrospectivaError,
)

_logger = get_logger(__name__)

# Mismo periodo que la linea base Seasonal Naive.
PERIODO_ESCALA = 7

_Pares = tuple[np.ndarray, np.ndarray]


def separar_reserva(
    serie: pd.Series, corte: ReserveCut
) -> tuple[np.ndarray, pd.Series]:
    """Historia <= t* (para la escala) y observaciones reales de la reserva."""
    dias = pd.to_datetime(pd.Series(serie.index)).dt.normalize()
    ordenada = pd.Series(
        serie.to_numpy(dtype=float), index=pd.DatetimeIndex(dias)
    ).sort_index()
    historia = ordenada.loc[ordenada.index <= corte.t_star].to_numpy()
    reserva = ordenada.loc[ordenada.index > corte.t_star]
    return historia, reserva


def escala_q1(
    historia: np.ndarray, m: int = PERIODO_ESCALA
) -> tuple[float | None, str | None]:
    y = np.asarray(historia, dtype=float)
    if len(y) <= m:
        return None, "historia_corta"
    q1 = float(np.mean(np.abs(y[m:] - y[:-m])))
    if not math.isfinite(q1):
        return None, "escala_no_finita"
    if q1 == 0.0:
        return None, "escala_cero"
    return q1, None


def evaluar_sku(entrada: EntradaSku, corte: ReserveCut) -> EvaluacionSku:
    """Metricas de la linea base y de cada candidato, por ventana y agregadas."""
    escala = escala_q1(entrada.historia)
    reales = entrada.reserva.dropna()
    base, ventanas_base = _evaluar(entrada.linea_base, entrada, corte, escala, None)
    candidatos = tuple(
        _evaluar(serie, entrada, corte, escala, ventanas_base)[0]
        for serie in entrada.candidatos
    )
    _logger.info(
        "Evaluacion sku=%s candidatos=%d ventanas_linea_base=%d/%d",
        entrada.sku,
        len(candidatos),
        base.n_ventanas_validas,
        base.n_ventanas_totales,
    )
    return EvaluacionSku(
        sku=entrada.sku,
        sku_class=entrada.sku_class,
        linea_base=base,
        candidatos=candidatos,
        n_obs_validas_reserva=int(np.isfinite(reales.to_numpy(dtype=float)).sum()),
        reserva_toda_cero=bool(len(reales) > 0 and np.all(reales.to_numpy() == 0)),
        n_candidatos_fallidos=entrada.n_candidatos_fallidos,
    )


def _evaluar(
    serie: SerieCandidato,
    entrada: EntradaSku,
    corte: ReserveCut,
    escala: tuple[float | None, str | None],
    ventanas_base: dict[pd.Timestamp, _Pares] | None,
) -> tuple[MetricasCandidato, dict[pd.Timestamp, _Pares]]:
    validas: dict[pd.Timestamp, _Pares] = {}
    vistos: set[pd.Timestamp] = set()
    n_pronosticos_validos = 0
    for pronostico in serie.pronosticos:
        _exigir_reserva(pronostico.origen, pronostico.fechas, serie, entrada, corte)
        if pronostico.origen in vistos:
            _rechazar(f"origen repetido {pronostico.origen.date()}", serie, entrada)
        vistos.add(pronostico.origen)
        y_pred = np.asarray(pronostico.valores, dtype=float)
        horizonte = len(pronostico.fechas)
        if (
            horizonte == 0
            or y_pred.shape != (horizonte,)
            or not np.all(np.isfinite(y_pred))
        ):
            continue
        n_pronosticos_validos += 1
        y_real = entrada.reserva.reindex(pronostico.fechas).to_numpy(dtype=float)
        if np.all(np.isfinite(y_real)):
            validas[pronostico.origen] = (y_real, y_pred)

    # La linea base se compara consigo misma: r = 1 (o no calculable).
    base = validas if ventanas_base is None else ventanas_base
    por_ventana = tuple(
        MetricasVentana(
            origen, _metricas(y_real, y_pred, escala, _comun(origen, base, y_pred))
        )
        for origen, (y_real, y_pred) in validas.items()
    )
    agregadas = None
    if validas:
        comunes = [o for o in validas if o in base]
        agregadas = _metricas(
            np.concatenate([v[0] for v in validas.values()]),
            np.concatenate([v[1] for v in validas.values()]),
            escala,
            (
                np.concatenate([validas[o][0] for o in comunes]),
                np.concatenate([validas[o][1] for o in comunes]),
                np.concatenate([base[o][1] for o in comunes]),
            )
            if comunes
            else None,
        )
    return (
        MetricasCandidato(
            candidato_id=serie.candidato_id,
            familia=serie.familia,
            modelo=serie.modelo,
            agregadas=agregadas,
            por_ventana=por_ventana,
            n_ventanas_totales=len(serie.pronosticos),
            n_ventanas_validas=len(validas),
            n_pronosticos_validos=n_pronosticos_validos,
        ),
        validas,
    )


def _comun(
    origen: pd.Timestamp, base: dict[pd.Timestamp, _Pares], y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if origen not in base:
        return None
    y_real, y_base = base[origen]
    return y_real, y_pred, y_base


def _metricas(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    escala: tuple[float | None, str | None],
    contra_base: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> Metricas:
    no_calculables: dict[str, str] = {}
    error_absoluto = mae(y_real, y_pred)

    q1, causa_q1 = escala
    mase = None
    if q1 is None:
        no_calculables["mase"] = causa_q1 or "escala_no_calculable"
    else:
        mase = error_absoluto / q1

    razon = None
    if contra_base is None:
        no_calculables["razon_sn"] = "sin_ventanas_comunes_con_linea_base"
    else:
        reales, pronostico, linea_base = contra_base
        rmse_base = rmse(reales, linea_base)
        if rmse_base == 0.0:
            no_calculables["razon_sn"] = "rmse_linea_base_cero"
        else:
            razon = rmse(reales, pronostico) / rmse_base

    return Metricas(
        n_pares=len(y_real),
        mae=error_absoluto,
        rmse=rmse(y_real, y_pred),
        me=float(np.mean(y_real - y_pred)),
        mase=mase,
        razon_sn=razon,
        no_calculables=no_calculables,
    )


def _exigir_reserva(
    origen: pd.Timestamp,
    fechas: pd.DatetimeIndex,
    serie: SerieCandidato,
    entrada: EntradaSku,
    corte: ReserveCut,
) -> None:
    # Causalidad: solo se evalua sobre la reserva, nunca sobre la historia de M2.
    if origen < corte.t_star or bool((fechas <= corte.t_star).any()):
        _rechazar(
            f"pronostico con origen {origen.date()} o fechas no posteriores a "
            f"t*={corte.t_star.date()}",
            serie,
            entrada,
        )


def _rechazar(motivo: str, serie: SerieCandidato, entrada: EntradaSku) -> None:
    _logger.error(
        "Evaluacion rechazada sku=%s candidato_id=%s: %s",
        entrada.sku,
        serie.candidato_id,
        motivo,
    )
    raise EvaluacionRetrospectivaError(
        f"sku={entrada.sku} candidato_id={serie.candidato_id}: {motivo}"
    )
