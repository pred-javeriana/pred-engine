"""Fase 0 de simulacion de pre-ingesta: aumentacion de datos y handoff al Modulo 1.

Submodulos:

* ``mbb`` - Moving Block Bootstrap sobre residuales STL.
* ``restricciones`` - compuerta de restricciones fisicas (0.3-A).
* ``divergencia`` - divergencia parametrica (0.3-B1).
* ``rechazo`` - bucle de rechazo y remuestreo (0.3-B2).
* ``contrato`` - contrato de datos de salida (0.4-A1).
* ``conformidad`` - validador de conformidad de esquema (0.4-A2).
* ``worm`` - guarda de escritura WORM (0.4-B1).
* ``exportador_csv`` - exportador del artefacto final (0.4-B2).
* ``fase0`` - orquestador One-Shot de la Fase 0 (0.4-C1).
* ``bitacora`` - bitacora estructurada de la corrida de handoff (0.4-C2).
"""

from pred_engine.aumentacion.bitacora import BitacoraCorrida, persistir_bitacora
from pred_engine.aumentacion.conformidad import (
    ReporteConformidad,
    validar_conformidad_o_fallar,
    verificar_conformidad,
)
from pred_engine.aumentacion.contrato import (
    CONTRACT_VERSION,
    OUTPUT_COLUMNS,
    OUTPUT_CONTRACT,
    describir_contrato,
)
from pred_engine.aumentacion.divergencia import (
    VeredictoDivergencia,
    evaluar_divergencia_parametrica,
)
from pred_engine.aumentacion.errores import (
    DivergenceRejectionExhausted,
    PhysicalConstraintError,
    SchemaConformanceError,
    WormOverwriteError,
)
from pred_engine.aumentacion.exportador_csv import (
    ArtefactoExportado,
    exportar_artefacto_csv,
)
from pred_engine.aumentacion.fase0 import (
    ConfiguracionCorrida,
    ResultadoFase0,
    ejecutar_fase_0,
)
from pred_engine.aumentacion.rechazo import (
    ResultadoRechazo,
    generar_series_aceptadas,
)
from pred_engine.aumentacion.restricciones import (
    LimitesLeadTime,
    acotar_lead_time,
    aplicar_restricciones_fisicas,
    limites_lead_time_desde_semilla,
    rectificar_demanda_no_negativa,
    truncar_a_unidades_enteras,
)
from pred_engine.aumentacion.worm import (
    escribir_una_sola_vez,
    resolver_ruta_artefacto,
)

__all__ = [
    "CONTRACT_VERSION",
    "OUTPUT_COLUMNS",
    "OUTPUT_CONTRACT",
    "ArtefactoExportado",
    "BitacoraCorrida",
    "ConfiguracionCorrida",
    "DivergenceRejectionExhausted",
    "LimitesLeadTime",
    "PhysicalConstraintError",
    "ReporteConformidad",
    "ResultadoFase0",
    "ResultadoRechazo",
    "SchemaConformanceError",
    "VeredictoDivergencia",
    "WormOverwriteError",
    "acotar_lead_time",
    "aplicar_restricciones_fisicas",
    "describir_contrato",
    "ejecutar_fase_0",
    "escribir_una_sola_vez",
    "evaluar_divergencia_parametrica",
    "exportar_artefacto_csv",
    "generar_series_aceptadas",
    "limites_lead_time_desde_semilla",
    "persistir_bitacora",
    "rectificar_demanda_no_negativa",
    "resolver_ruta_artefacto",
    "truncar_a_unidades_enteras",
    "validar_conformidad_o_fallar",
    "verificar_conformidad",
]
