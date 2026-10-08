"""Prueba Diebold-Mariano (HLN) del campeon contra Seasonal Naive (3.4).

Lee las metricas por ventana de `calculo_errores`; no recalcula errores. Se
aplica solo con datos reales y con correccion BH entre SKUs (ADR-03-008, alt. 2).
"""

from pred_engine.forecasting.evaluaciones.diebold_mariano.prueba import (
    PruebaDM,
    corregir_multiplicidad,
    probar_contra_linea_base,
    prueba_hln,
)

__all__ = [
    "PruebaDM",
    "corregir_multiplicidad",
    "probar_contra_linea_base",
    "prueba_hln",
]
