"""Motor de HPO agnostico de familia (TPE + ASHA sobre Walk-Forward Validation).

Punto de entrada unico: `ejecutar_estudio`. Las familias clasica, ML y DL
solo cambian `EspacioBusqueda` y `FabricaPronosticador`; el resto del motor
(muestreo, poda, asignacion de recursos, registro) es compartido.
"""

from pred_engine.optimizacion.optimizadores.HPO.asha import (
    AsignadorRecursosASHA,
    Decision,
)
from pred_engine.optimizacion.optimizadores.HPO.errores import (
    EspacioInvalidoError,
    EstudioError,
)
from pred_engine.optimizacion.optimizadores.HPO.espacio import (
    Categorico,
    Condicion,
    Entero,
    EspacioBusqueda,
    Flotante,
    Parametro,
)
from pred_engine.optimizacion.optimizadores.HPO.estudio import ejecutar_estudio
from pred_engine.optimizacion.optimizadores.HPO.muestreadores import (
    Muestreador,
    MuestreadorAleatorio,
    MuestreadorTPE,
)
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda, es_degenerada
from pred_engine.optimizacion.optimizadores.HPO.registro import RegistroEstudio

__all__ = [
    "AsignadorRecursosASHA",
    "Categorico",
    "Condicion",
    "Decision",
    "Entero",
    "EspacioBusqueda",
    "EspacioInvalidoError",
    "EstudioError",
    "Flotante",
    "Muestreador",
    "MuestreadorAleatorio",
    "MuestreadorTPE",
    "Parametro",
    "ReglasPoda",
    "RegistroEstudio",
    "ejecutar_estudio",
    "es_degenerada",
]
