"""Calculo de errores sobre la reserva (3.4-A1, ADR-03-006).

Contrasta pronosticos fechados contra las observaciones reales de la reserva y
contra Seasonal Naive, que genera en las mismas ventanas. Entrega la reserva
completa y sus bloques de seleccion y de prueba (ADR-023). No ajusta los
candidatos, no selecciona (veredictos) y no persiste (3.5).
"""

from pred_engine.forecasting.evaluaciones.calculo_errores.bloques import (
    Bloque,
    BloquesReserva,
)
from pred_engine.forecasting.evaluaciones.calculo_errores.contratos import (
    FAMILIA_LINEA_BASE,
    MODELO_LINEA_BASE,
    VERSION_METRICAS,
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
from pred_engine.forecasting.evaluaciones.calculo_errores.metricas import (
    PERIODO_ESCALA,
    escala_q1,
    evaluar_bloques,
    evaluar_sku,
    pares_comunes,
    pronosticar_linea_base,
    resumir_ventanas,
    separar_reserva,
)

__all__ = [
    "Bloque",
    "BloquesReserva",
    "EntradaSku",
    "EvaluacionBloques",
    "EvaluacionRetrospectivaError",
    "EvaluacionSku",
    "FAMILIA_LINEA_BASE",
    "MODELO_LINEA_BASE",
    "Metricas",
    "MetricasCandidato",
    "MetricasVentana",
    "PERIODO_ESCALA",
    "PronosticoFechado",
    "SerieCandidato",
    "VERSION_METRICAS",
    "escala_q1",
    "evaluar_bloques",
    "evaluar_sku",
    "pares_comunes",
    "pronosticar_linea_base",
    "resumir_ventanas",
    "separar_reserva",
]
