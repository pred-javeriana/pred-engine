"""Seleccion por categoria y veredictos por SKU (3.4-A2, ADR-03-007/008).

Lee solo las metricas de `calculo_errores`: no recalcula errores ni escribe
archivos (eso es de 3.5).
"""

from pred_engine.forecasting.evaluaciones.veredictos.contratos import (
    FAMILIAS_ELEGIBLES_INICIALES,
    POLITICA_INICIAL,
    VEREDICTOS,
    ModeloEvaluado,
    MotivoSeleccion,
    PoliticaSeleccion,
    ResultadoEvaluacion,
    ResumenCategoria,
    SeleccionCategoria,
    Veredicto,
    VeredictoSku,
)
from pred_engine.forecasting.evaluaciones.veredictos.seleccion import (
    representante,
    seleccionar_categoria,
)
from pred_engine.forecasting.evaluaciones.veredictos.veredictos import (
    emitir_veredictos,
)

__all__ = [
    "FAMILIAS_ELEGIBLES_INICIALES",
    "ModeloEvaluado",
    "MotivoSeleccion",
    "POLITICA_INICIAL",
    "PoliticaSeleccion",
    "ResultadoEvaluacion",
    "ResumenCategoria",
    "SeleccionCategoria",
    "VEREDICTOS",
    "Veredicto",
    "VeredictoSku",
    "emitir_veredictos",
    "representante",
    "seleccionar_categoria",
]
