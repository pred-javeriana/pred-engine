"""Calculo de errores sobre la reserva (3.4-A1, ADR-03-006).

Contrasta pronosticos fechados contra las observaciones reales de la reserva y
contra Seasonal Naive. No ajusta modelos, no selecciona (veredictos) y no
persiste (3.5).
"""

from pred_engine.forecasting.evaluaciones.calculo_errores.contratos import (
    FAMILIA_LINEA_BASE,
    VERSION_METRICAS,
    EntradaSku,
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
from pred_engine.forecasting.evaluaciones.calculo_errores.metricas import (
    PERIODO_ESCALA,
    escala_q1,
    evaluar_sku,
    separar_reserva,
)

__all__ = [
    "EntradaSku",
    "EvaluacionRetrospectivaError",
    "EvaluacionSku",
    "FAMILIA_LINEA_BASE",
    "Metricas",
    "MetricasCandidato",
    "MetricasVentana",
    "PERIODO_ESCALA",
    "PronosticoFechado",
    "SerieCandidato",
    "VERSION_METRICAS",
    "escala_q1",
    "evaluar_sku",
    "separar_reserva",
]
