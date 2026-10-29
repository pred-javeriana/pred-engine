"""Constructores de evidencia para las pruebas de 3.4 (metricas y veredictos).

No empieza con `test_`: pytest no lo recolecta; lo importan los `test_*.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from pred_engine.comun.reserva import ReserveCut
from pred_engine.forecasting.evaluaciones.calculo_errores import (
    FAMILIA_LINEA_BASE,
    BloquesReserva,
    EvaluacionBloques,
    EvaluacionSku,
    Metricas,
    MetricasCandidato,
    MetricasVentana,
    PronosticoFechado,
    SerieCandidato,
)
from pred_engine.forecasting.evaluaciones.veredictos import POLITICA_INICIAL

T_ESTRELLA = pd.Timestamp("2024-01-14")


def corte(dias_reservados: int = 40) -> ReserveCut:
    return ReserveCut(
        t_star=T_ESTRELLA,
        first_reserved=T_ESTRELLA + pd.Timedelta(days=1),
        last_observed=T_ESTRELLA + pd.Timedelta(days=dias_reservados),
        reserved_days=dias_reservados,
    )


def fechas_desde(origen: pd.Timestamp, horizonte: int) -> pd.DatetimeIndex:
    return pd.date_range(origen + pd.Timedelta(days=1), periods=horizonte, freq="D")


def pronostico(dias_tras_corte: int, valores: Sequence[float]) -> PronosticoFechado:
    """Pronostico con origen `t* + dias_tras_corte` para los dias siguientes."""
    origen = T_ESTRELLA + pd.Timedelta(days=dias_tras_corte)
    return PronosticoFechado(
        origen=origen,
        fechas=fechas_desde(origen, len(valores)),
        valores=np.asarray(valores, dtype=float),
    )


def serie(
    candidato_id: str,
    pronosticos: Sequence[PronosticoFechado],
    familia: str = "classical",
    modelo: str = "sarima",
) -> SerieCandidato:
    return SerieCandidato(candidato_id, familia, modelo, tuple(pronosticos))


def reserva(valores: Sequence[float] | np.ndarray) -> pd.Series:
    return pd.Series(
        np.asarray(valores, dtype=float), index=fechas_desde(T_ESTRELLA, len(valores))
    )


# --- Evidencia ya calculada, para probar seleccion y veredictos ----------


def metricas(
    r: float | None, rmse: float = 1.0, mae: float = 1.0, n_pares: int = 1
) -> Metricas:
    return Metricas(
        n_pares=n_pares,
        mae=mae,
        rmse=rmse,
        me=0.0,
        mase=None,
        razon_sn=r,
        no_calculables={} if r is not None else {"razon_sn": "prueba"},
    )


def metricas_candidato(
    familia: str,
    r: float | None,
    *,
    rmse: float = 1.0,
    n: int = 40,
    maes: Sequence[float] | None = None,
    rmses: Sequence[float] | None = None,
    pronosticos_validos: int | None = None,
    candidato_id: str | None = None,
    horizonte: int = 1,
) -> MetricasCandidato:
    maes = list(maes) if maes is not None else [1.0] * n
    rmses = list(rmses) if rmses is not None else [1.0] * n
    return MetricasCandidato(
        candidato_id=candidato_id or f"S/{familia}",
        familia=familia,
        modelo=familia,
        agregadas=metricas(r, rmse=rmse) if n else None,
        por_ventana=tuple(
            MetricasVentana(
                T_ESTRELLA + pd.Timedelta(days=i),
                metricas(r, rmse=e, mae=m, n_pares=horizonte),
            )
            for i, (m, e) in enumerate(zip(maes[:n], rmses[:n], strict=True))
        ),
        n_ventanas_totales=n,
        n_ventanas_validas=n,
        n_pronosticos_validos=n if pronosticos_validos is None else pronosticos_validos,
    )


def par_fieller_disjunto() -> tuple[MetricasCandidato, MetricasCandidato]:
    """Campeon y Seasonal Naive cuyo intervalo de r al 95 % tiene dos tramos.

    40 ventanas de horizonte 7. La perdida de Seasonal Naive es 1 salvo siete
    ventanas de 100, asi que no se distingue de cero; la del campeon es 0,8
    menor con un ruido pequeno. HLN-DM rechaza r = 1 con p del orden de 1e-87.
    """
    linea_base = np.ones(40)
    linea_base[15:22] = 100
    campeon = linea_base - 0.8 + 0.01 * np.sin(np.arange(40))
    return (
        metricas_candidato("ml", None, rmses=list(np.sqrt(campeon)), horizonte=7),
        metricas_candidato(
            FAMILIA_LINEA_BASE, None, rmses=list(np.sqrt(linea_base)), horizonte=7
        ),
    )


def evaluacion(
    sku: str,
    candidatos: Sequence[MetricasCandidato],
    *,
    sku_class: str = "smooth",
    n_obs: int = 40,
    toda_cero: bool = False,
    fallidos: int = 0,
    base: MetricasCandidato | None = None,
) -> EvaluacionSku:
    return EvaluacionSku(
        sku=sku,
        sku_class=sku_class,  # type: ignore[arg-type]
        linea_base=base or metricas_candidato(FAMILIA_LINEA_BASE, 1.0),
        candidatos=tuple(candidatos),
        n_obs_validas_reserva=n_obs,
        reserva_toda_cero=toda_cero,
        n_candidatos_fallidos=fallidos,
    )


# Reserva de 100 dias con horizonte 1: seleccion con los origenes t*..t*+39 y
# prueba con t*+40..t*+99, sin embargo.
BLOQUES = BloquesReserva(T_ESTRELLA, 100, 1, POLITICA_INICIAL.fraccion_seleccion)


def en_bloques(
    prueba: EvaluacionSku, seleccion: EvaluacionSku | None = None
) -> EvaluacionBloques:
    """La evidencia de `prueba` en el bloque de prueba de `BLOQUES`.

    Las ventanas de `metricas_candidato` empiezan en t*; aqui se corren al
    primer origen de prueba. Sin `seleccion`, el bloque de seleccion tiene la
    misma evidencia en sus propios origenes. Los veredictos no leen `reserva`.
    """
    corrida = _desplazar(prueba, BLOQUES.dias_seleccion)
    return EvaluacionBloques(
        reserva=corrida, seleccion=seleccion or prueba, prueba=corrida
    )


def _desplazar(evaluado: EvaluacionSku, dias: int) -> EvaluacionSku:
    def mover(c: MetricasCandidato) -> MetricasCandidato:
        return replace(
            c,
            por_ventana=tuple(
                replace(v, origen=v.origen + pd.Timedelta(days=dias))
                for v in c.por_ventana
            ),
        )

    return replace(
        evaluado,
        linea_base=mover(evaluado.linea_base),
        candidatos=tuple(mover(c) for c in evaluado.candidatos),
    )
