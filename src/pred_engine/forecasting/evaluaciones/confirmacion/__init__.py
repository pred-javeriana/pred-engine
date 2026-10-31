"""Confirmacion del campeon congelado de cada categoria (ADR-023).

Lee solo el bloque de prueba. Compara el campeon con cada familia competidora
(Wilcoxon, interseccion-union y Holm) y resume los efectos con intervalos.
"""

from pred_engine.forecasting.evaluaciones.confirmacion.confirmacion import (
    confirmar_campeones,
)
from pred_engine.forecasting.evaluaciones.confirmacion.contratos import (
    ESTADOS_CONFIRMACION,
    CategoriaAConfirmar,
    ComparacionFamilia,
    ConfirmacionCategoria,
    EfectoLineaBase,
    EstadoConfirmacion,
    Intervalo,
    ParametrosConfirmacion,
)

__all__ = [
    "ESTADOS_CONFIRMACION",
    "CategoriaAConfirmar",
    "ComparacionFamilia",
    "ConfirmacionCategoria",
    "EfectoLineaBase",
    "EstadoConfirmacion",
    "Intervalo",
    "ParametrosConfirmacion",
    "confirmar_campeones",
]
