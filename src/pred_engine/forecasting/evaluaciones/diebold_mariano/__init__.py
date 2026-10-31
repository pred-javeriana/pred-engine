"""Prueba Diebold-Mariano (HLN) del campeon contra Seasonal Naive (3.4).

Lee las metricas por ventana de `calculo_errores`; no recalcula errores. Se
aplica solo con datos reales, con correccion BH entre los SKU de cada
categoria y con el intervalo de r que invierte la misma prueba (ADR-03-011,
ADR-023).
"""

from pred_engine.forecasting.evaluaciones.diebold_mariano.prueba import (
    IntervaloRazon,
    PruebaDM,
    corregir_multiplicidad,
    intervalo_razon,
    probar_contra_linea_base,
    probar_pares,
    prueba_hln,
)

__all__ = [
    "IntervaloRazon",
    "PruebaDM",
    "corregir_multiplicidad",
    "intervalo_razon",
    "probar_contra_linea_base",
    "probar_pares",
    "prueba_hln",
]
