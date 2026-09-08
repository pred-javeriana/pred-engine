"""Dataclasses compartidas por ingesta, optimizacion y forecasting."""

from pred_engine.comun.dataclasses.hpo import ResultadoEstudio, Trial
from pred_engine.comun.dataclasses.modelos_clasicos import (
    ConfiguracionSeleccionada,
    ResultadoSeleccionClasica,
)
from pred_engine.comun.dataclasses.validacion_temporal import (
    EstadoParcial,
    ResultadoVentana,
    ResultadoWalkForward,
    VentanaTemporal,
)

__all__ = [
    "ConfiguracionSeleccionada",
    "EstadoParcial",
    "ResultadoEstudio",
    "ResultadoSeleccionClasica",
    "ResultadoVentana",
    "ResultadoWalkForward",
    "Trial",
    "VentanaTemporal",
]
