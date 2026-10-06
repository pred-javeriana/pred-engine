"""Constructores de evidencia para las pruebas de 3.4 (metricas y veredictos).

No empieza con `test_`: pytest no lo recolecta; lo importan los `test_*.py`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from pred_engine.comun.reserva import ReserveCut
from pred_engine.forecasting.evaluaciones.calculo_errores import (
    FAMILIA_LINEA_BASE,
    EvaluacionSku,
    Metricas,
    MetricasCandidato,
    MetricasVentana,
    PronosticoFechado,
    SerieCandidato,
)

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


def linea_base(pronosticos: Sequence[PronosticoFechado]) -> SerieCandidato:
    return serie("S1/linea_base", pronosticos, FAMILIA_LINEA_BASE, "seasonal_naive")


def reserva(valores: Sequence[float]) -> pd.Series:
    return pd.Series(
        np.asarray(valores, dtype=float), index=fechas_desde(T_ESTRELLA, len(valores))
    )


# --- Evidencia ya calculada, para probar seleccion y veredictos ----------


def metricas(r: float | None, rmse: float = 1.0, mae: float = 1.0) -> Metricas:
    return Metricas(
        n_pares=1,
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
    pronosticos_validos: int | None = None,
    candidato_id: str | None = None,
) -> MetricasCandidato:
    maes = list(maes) if maes is not None else [1.0] * n
    return MetricasCandidato(
        candidato_id=candidato_id or f"S/{familia}",
        familia=familia,
        modelo=familia,
        agregadas=metricas(r, rmse=rmse) if n else None,
        por_ventana=tuple(
            MetricasVentana(T_ESTRELLA + pd.Timedelta(days=i), metricas(r, mae=m))
            for i, m in enumerate(maes[:n])
        ),
        n_ventanas_totales=n,
        n_ventanas_validas=n,
        n_pronosticos_validos=n if pronosticos_validos is None else pronosticos_validos,
    )


def evaluacion(
    sku: str,
    candidatos: Sequence[MetricasCandidato],
    *,
    sku_class: str = "lumpy",
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
