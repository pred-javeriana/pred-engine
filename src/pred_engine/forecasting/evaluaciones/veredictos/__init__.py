"""Seleccion por categoria y veredictos por SKU (3.4-A2, ADR-03-007/008, ADR-023).

Lee solo las metricas de `calculo_errores`: la seleccion, las del bloque de
seleccion; los veredictos y las pruebas, las del bloque de prueba. No recalcula
errores ni escribe archivos (eso es de 3.5).
"""

from pred_engine.forecasting.evaluaciones.veredictos.contratos import (
    FAMILIA_FUNDACIONAL,
    FAMILIAS_ELEGIBLES_INICIALES,
    POLITICA_INICIAL,
    VEREDICTOS,
    ModeloEvaluado,
    MotivoSeleccion,
    PoliticaSeleccion,
    ResultadoEvaluacion,
    ResumenCategoria,
    SeleccionCategoria,
    SignificanciaCategoria,
    Veredicto,
    VeredictoSku,
)
from pred_engine.forecasting.evaluaciones.veredictos.seleccion import (
    representante,
    seleccionar_categoria,
)
from pred_engine.forecasting.evaluaciones.veredictos.seleccion_previa import (
    CandidatoPrevio,
    evaluacion_previa,
    seleccionar_previa,
)
from pred_engine.forecasting.evaluaciones.veredictos.veredictos import (
    emitir_veredictos,
)

__all__ = [
    "FAMILIA_FUNDACIONAL",
    "FAMILIAS_ELEGIBLES_INICIALES",
    "CandidatoPrevio",
    "ModeloEvaluado",
    "MotivoSeleccion",
    "POLITICA_INICIAL",
    "PoliticaSeleccion",
    "ResultadoEvaluacion",
    "ResumenCategoria",
    "SeleccionCategoria",
    "SignificanciaCategoria",
    "VEREDICTOS",
    "Veredicto",
    "VeredictoSku",
    "emitir_veredictos",
    "evaluacion_previa",
    "representante",
    "seleccionar_categoria",
    "seleccionar_previa",
]
