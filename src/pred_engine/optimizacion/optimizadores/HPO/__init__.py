"""Motor de HPO agnostico de familia (TPE + ASHA sobre Walk-Forward Validation).

Punto de entrada unico: `ejecutar_estudio`. Las familias clasica, ML y DL
solo cambian `EspacioBusqueda` y `FabricaPronosticador`; el resto del motor
(muestreo, poda, asignacion de recursos, registro) es compartido.
"""

from pred_engine.optimizacion.optimizadores.HPO.asha import PodadorASHA
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
    construir_muestreador_aleatorio,
    construir_muestreador_tpe,
)
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda, es_degenerada
from pred_engine.optimizacion.optimizadores.HPO.registro import (
    instantanea_desde_estudio,
    reanudar_estudio,
    volcar_jsonl,
)

__all__ = [
    "Categorico",
    "Condicion",
    "Entero",
    "EspacioBusqueda",
    "EspacioInvalidoError",
    "EstudioError",
    "Flotante",
    "Parametro",
    "PodadorASHA",
    "ReglasPoda",
    "construir_muestreador_aleatorio",
    "construir_muestreador_tpe",
    "ejecutar_estudio",
    "es_degenerada",
    "instantanea_desde_estudio",
    "reanudar_estudio",
    "volcar_jsonl",
]
