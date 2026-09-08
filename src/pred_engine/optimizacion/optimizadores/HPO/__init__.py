"""Motor de HPO agnostico de familia (TPE + ASHA sobre Walk-Forward Validation).

Punto de entrada unico: `ejecutar_estudio`. Las familias clasica, ML y DL
solo cambian `EspacioBusqueda` y `FabricaPronosticador`; el resto del motor
(muestreo, poda, asignacion de recursos, registro) es compartido.

`estudio.py`/`asha.py`/`registro.py` no importan Optuna directamente: hablan
`contratos.py`. Quien conoce Optuna es `adaptador_optuna.py`.
"""

from pred_engine.optimizacion.optimizadores.HPO.adaptador_optuna import (
    crear_estudio_optuna,
    reanudar_estudio,
    volcar_jsonl,
)
from pred_engine.optimizacion.optimizadores.HPO.asha import DecisionPoda, DecisorASHA
from pred_engine.optimizacion.optimizadores.HPO.contratos import (
    EstadoTrialBackend,
    EstudioHPO,
    InfoTrial,
    ProveedorMotivoPoda,
    TrialHPO,
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
    construir_muestreador_aleatorio,
    construir_muestreador_tpe,
)
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda, es_degenerada
from pred_engine.optimizacion.optimizadores.HPO.registro import (
    instantanea_desde_estudio,
)

__all__ = [
    "Categorico",
    "Condicion",
    "DecisionPoda",
    "DecisorASHA",
    "Entero",
    "EspacioBusqueda",
    "EspacioInvalidoError",
    "EstadoTrialBackend",
    "EstudioError",
    "EstudioHPO",
    "Flotante",
    "InfoTrial",
    "Parametro",
    "ProveedorMotivoPoda",
    "ReglasPoda",
    "TrialHPO",
    "construir_muestreador_aleatorio",
    "construir_muestreador_tpe",
    "crear_estudio_optuna",
    "ejecutar_estudio",
    "es_degenerada",
    "instantanea_desde_estudio",
    "reanudar_estudio",
    "volcar_jsonl",
]
