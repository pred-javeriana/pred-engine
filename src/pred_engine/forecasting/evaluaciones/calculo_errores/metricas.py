"""Metricas de error sobre la reserva (3.4-A1, ADR-03-006).

De los candidatos solo recibe arreglos y fechas: no puede reajustarlos. El
unico modelo que ajusta es la linea base Seasonal Naive (3.4-A1, paso 2).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pred_engine.comun.logger import get_logger
from pred_engine.comun.reserva import ReserveCut
from pred_engine.comun.walkforward.metricas import mae, rmse
from pred_engine.forecasting.adaptador_candidatos import instanciar_linea_base
from pred_engine.forecasting.evaluaciones.calculo_errores.bloques import (
    BloquesReserva,
)
from pred_engine.forecasting.evaluaciones.calculo_errores.contratos import (
    FAMILIA_LINEA_BASE,
    MODELO_LINEA_BASE,
    EntradaSku,
    EvaluacionBloques,
    EvaluacionSku,
    Metricas,
    MetricasCandidato,
    MetricasVentana,
    PronosticoFechado,
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


def pronosticar_linea_base(
    entrada: EntradaSku,
    corte: ReserveCut,
    *,
    ventanas_planeadas: Mapping[pd.Timestamp, pd.DatetimeIndex] | None = None,
    confirmados: Sequence[PronosticoFechado] = (),
) -> SerieCandidato:
    """Seasonal Naive en las mismas ventanas que los candidatos (3.4-A1, ADR-03-004).

    Se ajusta una vez con la historia <= t* y pronostica desde cada origen con lo
    observado en la reserva hasta ese origen, sin reestimar. Si la historia es
    mas corta que el periodo o falta una observacion hasta el origen, esa
    ventana queda sin valores finitos y se cuenta como invalida.
    """
    ventanas: dict[pd.Timestamp, pd.DatetimeIndex] = dict(ventanas_planeadas or {})
    for serie in entrada.candidatos:
        for pronostico in serie.pronosticos:
            origen, fechas = pronostico.origen, pronostico.fechas
            _exigir_reserva(origen, fechas, serie, entrada, corte)
            siguientes = pd.date_range(
                origen + pd.Timedelta(days=1), periods=len(fechas), freq="D"
            )
            if not (
                fechas.equals(siguientes)
                and fechas.equals(ventanas.setdefault(origen, fechas))
            ):
                _rechazar(
                    f"la ventana con origen {origen.date()} no son los dias que "
                    "siguen al origen o difiere entre candidatos",
                    serie,
                    entrada,
                )

    restaurados = {p.origen: p for p in confirmados}
    if len(restaurados) != len(confirmados) or any(
        p.origen not in ventanas
        or not p.fechas.equals(ventanas[p.origen])
        or np.asarray(p.valores).shape != (len(p.fechas),)
        for p in confirmados
    ):
        raise EvaluacionRetrospectivaError(
            "linea base confirmada incompatible con las ventanas"
        )
    modelo = instanciar_linea_base()
    ajustado = False
    if len(restaurados) < len(ventanas):
        try:
            modelo.fit(entrada.historia)
            ajustado = True
        except ValueError:
            _logger.warning(
                "Linea base sin ajustar sku=%s: historia de %d dias",
                entrada.sku,
                len(entrada.historia),
            )
    pronosticos = []
    for origen in sorted(ventanas):
        if origen in restaurados:
            pronosticos.append(restaurados[origen])
            continue
        fechas = ventanas[origen]
        observadas = entrada.reserva.reindex(
            pd.date_range(corte.first_reserved, origen, freq="D")
        ).to_numpy(dtype=float)
        valores = np.full(len(fechas), np.nan)
        if ajustado and len(fechas) and np.all(np.isfinite(observadas)):
            valores = modelo.pronosticar(observadas, len(fechas))
        pronosticos.append(PronosticoFechado(origen, fechas, valores))
    return SerieCandidato(
        candidato_id=f"{entrada.sku}/{FAMILIA_LINEA_BASE}/{MODELO_LINEA_BASE}",
        familia=FAMILIA_LINEA_BASE,
        modelo=MODELO_LINEA_BASE,
        pronosticos=tuple(pronosticos),
    )


def evaluar_sku(
    entrada: EntradaSku, corte: ReserveCut, *, linea_base: SerieCandidato | None = None
) -> EvaluacionSku:
    """Metricas de la linea base y de cada candidato en toda la reserva."""
    return _Evaluacion(entrada, corte, linea_base).vista(entrada.reserva.index, None)


def evaluar_bloques(
    entrada: EntradaSku,
    corte: ReserveCut,
    bloques: BloquesReserva,
    *,
    linea_base: SerieCandidato | None = None,
) -> EvaluacionBloques:
    """Metricas en toda la reserva y en cada bloque, con una sola pasada (ADR-023)."""
    if (bloques.t_star, bloques.dias_reserva) != (corte.t_star, corte.reserved_days):
        raise EvaluacionRetrospectivaError("bloques de otra reserva")
    evaluacion = _Evaluacion(entrada, corte, linea_base)
    return EvaluacionBloques(
        reserva=evaluacion.vista(entrada.reserva.index, None),
        seleccion=evaluacion.vista(
            bloques.dias("seleccion"), lambda o: bloques.bloque(o) == "seleccion"
        ),
        prueba=evaluacion.vista(
            bloques.dias("prueba"), lambda o: bloques.bloque(o) == "prueba"
        ),
    )


def pares_comunes(
    candidato: MetricasCandidato, linea_base: MetricasCandidato
) -> list[tuple[Metricas, Metricas]]:
    """Metricas de las ventanas validas para ambos, en el orden del candidato."""
    base = {v.origen: v.metricas for v in linea_base.por_ventana}
    return [
        (v.metricas, base[v.origen]) for v in candidato.por_ventana if v.origen in base
    ]


@dataclass(frozen=True, slots=True, eq=False)
class _Ventanas:
    """Pronosticos de una serie ya validados contra la reserva."""

    serie: SerieCandidato
    origenes: tuple[pd.Timestamp, ...]
    pronosticos_validos: frozenset[pd.Timestamp]
    validas: dict[pd.Timestamp, _Pares]


class _Evaluacion:
    """Una pasada por los pronosticos del SKU; cada vista filtra sus ventanas."""

    def __init__(
        self,
        entrada: EntradaSku,
        corte: ReserveCut,
        linea_base: SerieCandidato | None,
    ) -> None:
        self.entrada = entrada
        self.escala = escala_q1(entrada.historia)
        if linea_base is None:
            linea_base = pronosticar_linea_base(entrada, corte)
        elif (
            linea_base.candidato_id
            != f"{entrada.sku}/{FAMILIA_LINEA_BASE}/{MODELO_LINEA_BASE}"
            or linea_base.familia != FAMILIA_LINEA_BASE
        ):
            raise EvaluacionRetrospectivaError("identidad de linea base incompatible")
        else:
            # Valida el calendario sin volver a emitir pronosticos confirmados.
            linea_base = pronosticar_linea_base(
                entrada,
                corte,
                ventanas_planeadas={p.origen: p.fechas for p in linea_base.pronosticos},
                confirmados=linea_base.pronosticos,
            )
        self.base = _ventanas(linea_base, entrada, corte)
        # La linea base se compara consigo misma: r = 1 (o no calculable).
        self.series = tuple(
            (ventanas, _resumen(ventanas, self.escala, self.base.validas))
            for ventanas in (
                self.base,
                *(_ventanas(serie, entrada, corte) for serie in entrada.candidatos),
            )
        )
        completa = self.series[0][1]
        _logger.info(
            "Evaluacion sku=%s candidatos=%d ventanas_linea_base=%d/%d",
            entrada.sku,
            len(entrada.candidatos),
            completa.n_ventanas_validas,
            completa.n_ventanas_totales,
        )

    def vista(
        self,
        dias: pd.Index,
        en_tramo: Callable[[pd.Timestamp], bool] | None,
    ) -> EvaluacionSku:
        entrada = self.entrada
        base, *candidatos = (
            self._metricas(ventanas, completa, en_tramo)
            for ventanas, completa in self.series
        )
        reales = entrada.reserva.reindex(dias).dropna()
        return EvaluacionSku(
            sku=entrada.sku,
            sku_class=entrada.sku_class,
            linea_base=base,
            candidatos=tuple(candidatos),
            n_obs_validas_reserva=int(np.isfinite(reales.to_numpy(dtype=float)).sum()),
            reserva_toda_cero=bool(len(reales) > 0 and np.all(reales.to_numpy() == 0)),
            n_candidatos_fallidos=entrada.n_candidatos_fallidos,
        )

    def _metricas(
        self,
        ventanas: _Ventanas,
        completa: MetricasCandidato,
        en_tramo: Callable[[pd.Timestamp], bool] | None,
    ) -> MetricasCandidato:
        if en_tramo is None:
            return completa
        validas = {o: p for o, p in ventanas.validas.items() if en_tramo(o)}
        return MetricasCandidato(
            candidato_id=completa.candidato_id,
            familia=completa.familia,
            modelo=completa.modelo,
            agregadas=_agregadas(validas, self.base.validas, self.escala),
            por_ventana=tuple(v for v in completa.por_ventana if en_tramo(v.origen)),
            n_ventanas_totales=sum(1 for o in ventanas.origenes if en_tramo(o)),
            n_ventanas_validas=len(validas),
            n_pronosticos_validos=sum(
                1 for o in ventanas.pronosticos_validos if en_tramo(o)
            ),
        )


def _ventanas(
    serie: SerieCandidato, entrada: EntradaSku, corte: ReserveCut
) -> _Ventanas:
    validas: dict[pd.Timestamp, _Pares] = {}
    origenes: dict[pd.Timestamp, None] = {}
    pronosticos_validos: set[pd.Timestamp] = set()
    for pronostico in serie.pronosticos:
        _exigir_reserva(pronostico.origen, pronostico.fechas, serie, entrada, corte)
        if pronostico.origen in origenes:
            _rechazar(f"origen repetido {pronostico.origen.date()}", serie, entrada)
        origenes[pronostico.origen] = None
        y_pred = np.asarray(pronostico.valores, dtype=float)
        horizonte = len(pronostico.fechas)
        if (
            horizonte == 0
            or y_pred.shape != (horizonte,)
            or not np.all(np.isfinite(y_pred))
        ):
            continue
        pronosticos_validos.add(pronostico.origen)
        y_real = entrada.reserva.reindex(pronostico.fechas).to_numpy(dtype=float)
        if np.all(np.isfinite(y_real)):
            validas[pronostico.origen] = (y_real, y_pred)
    return _Ventanas(serie, tuple(origenes), frozenset(pronosticos_validos), validas)


def resumir_ventanas(
    serie: SerieCandidato,
    validas: Mapping[pd.Timestamp, tuple[np.ndarray, np.ndarray]],
    base: Mapping[pd.Timestamp, tuple[np.ndarray, np.ndarray]],
    escala: tuple[float | None, str | None],
    *,
    n_ventanas_totales: int,
    n_pronosticos_validos: int,
) -> MetricasCandidato:
    """Metricas por ventana y agregadas de pares (real, pronostico) por origen.

    ``base`` son los pares de la linea base; r usa solo las ventanas comunes y
    la linea base se pasa a si misma. ``serie`` solo aporta la identidad.
    """
    return MetricasCandidato(
        candidato_id=serie.candidato_id,
        familia=serie.familia,
        modelo=serie.modelo,
        agregadas=_agregadas(validas, base, escala),
        por_ventana=tuple(
            MetricasVentana(
                origen, _metricas(y_real, y_pred, escala, _comun(origen, base, y_pred))
            )
            for origen, (y_real, y_pred) in validas.items()
        ),
        n_ventanas_totales=n_ventanas_totales,
        n_ventanas_validas=len(validas),
        n_pronosticos_validos=n_pronosticos_validos,
    )


def _resumen(
    ventanas: _Ventanas,
    escala: tuple[float | None, str | None],
    base: dict[pd.Timestamp, _Pares],
) -> MetricasCandidato:
    return resumir_ventanas(
        ventanas.serie,
        ventanas.validas,
        base,
        escala,
        n_ventanas_totales=len(ventanas.origenes),
        n_pronosticos_validos=len(ventanas.pronosticos_validos),
    )


def _agregadas(
    validas: Mapping[pd.Timestamp, _Pares],
    base: Mapping[pd.Timestamp, _Pares],
    escala: tuple[float | None, str | None],
) -> Metricas | None:
    if not validas:
        return None
    comunes = [o for o in validas if o in base]
    return _metricas(
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


def _comun(
    origen: pd.Timestamp, base: Mapping[pd.Timestamp, _Pares], y_pred: np.ndarray
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
