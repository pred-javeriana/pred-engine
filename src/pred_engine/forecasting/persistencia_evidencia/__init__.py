"""Persistencia y publicacion de la evidencia de M3 (3.5-A1).

Guarda en SQLite lo que producen 3.2 (fallos), 3.3 (pronosticos por ventana)
y 3.4 (metricas, seleccion y veredictos) con una maquina de estados por
corrida; al publicar deriva el Parquet. No evalua ni orquesta la reanudacion:
expone las unidades confirmadas para quien la coordine (3.0).
"""

from pred_engine.forecasting.persistencia_evidencia.contratos import (
    EstadoCorrida,
    IdentidadCorrida,
)
from pred_engine.forecasting.persistencia_evidencia.errores import (
    EstadoCorridaError,
    EvidenciaInconsistenteError,
    PersistenciaEvidenciaError,
)
from pred_engine.forecasting.persistencia_evidencia.repositorio import (
    TABLAS,
    TABLAS_POR_SKU,
    VENTANA_AGREGADA,
    consultar,
    esquema_m3,
    estado_corrida,
    guardar_evaluacion,
    guardar_unidades,
    marcar_evaluada,
    publicar,
    registrar_corrida,
    unidades_confirmadas,
)

__all__ = [
    "EstadoCorrida",
    "EstadoCorridaError",
    "EvidenciaInconsistenteError",
    "IdentidadCorrida",
    "PersistenciaEvidenciaError",
    "TABLAS",
    "TABLAS_POR_SKU",
    "VENTANA_AGREGADA",
    "consultar",
    "esquema_m3",
    "estado_corrida",
    "guardar_evaluacion",
    "guardar_unidades",
    "marcar_evaluada",
    "publicar",
    "registrar_corrida",
    "unidades_confirmadas",
]
