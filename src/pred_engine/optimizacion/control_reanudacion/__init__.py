"""Control de reanudacion del Modulo 2: ciclo de vida de corrida PRED.

Administra manifiesto, huella y recuperacion. No importa Optuna: el
estado de los trials vive en el adaptador del backend HPO.
"""

from pred_engine.optimizacion.control_reanudacion.almacenamiento import (
    AlmacenManifiestosFs,
    escribir_atomico,
)
from pred_engine.optimizacion.control_reanudacion.contratos import (
    SCHEMA_VERSION,
    AlmacenManifiestos,
    BackendHPO,
    ConfiguracionOptimizador,
    ConfiguracionValidacion,
    EstadoCorrida,
    ManifiestoCorrida,
    PuertoPersistenciaMotor,
    SolicitudCorrida,
)
from pred_engine.optimizacion.control_reanudacion.controlador import (
    ControladorReanudacion,
    SesionCorrida,
)
from pred_engine.optimizacion.control_reanudacion.errores import (
    CorridaFallidaError,
    IncompatibilidadCorridaError,
    ManifiestoAusenteError,
    ManifiestoCorruptoError,
    ReanudacionError,
    TransicionEstadoError,
    VersionManifiestoError,
)
from pred_engine.optimizacion.control_reanudacion.huella import (
    calcular_huella,
    verificar_compatibilidad,
)
from pred_engine.optimizacion.control_reanudacion.transiciones import (
    TRANSICIONES_PERMITIDAS,
    transicionar,
)

__all__ = [
    "SCHEMA_VERSION",
    "AlmacenManifiestos",
    "AlmacenManifiestosFs",
    "BackendHPO",
    "ConfiguracionOptimizador",
    "ConfiguracionValidacion",
    "calcular_huella",
    "ControladorReanudacion",
    "CorridaFallidaError",
    "escribir_atomico",
    "EstadoCorrida",
    "IncompatibilidadCorridaError",
    "ManifiestoAusenteError",
    "ManifiestoCorruptoError",
    "ManifiestoCorrida",
    "PuertoPersistenciaMotor",
    "ReanudacionError",
    "SesionCorrida",
    "SolicitudCorrida",
    "TRANSICIONES_PERMITIDAS",
    "TransicionEstadoError",
    "verificar_compatibilidad",
    "VersionManifiestoError",
    "transicionar",
]
