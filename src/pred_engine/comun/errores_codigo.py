"""Codigo estable de cada error del motor (seccion 11 del contrato, G6 y G7).

El codigo se deduce del *tipo* de la excepcion, nunca de su mensaje. Las clases se
resuelven de forma perezosa (modulo, nombre) para que este modulo, que vive en
``comun``, no importe las capas superiores al cargarse ni cree ciclos; tambien lo
usan los subprocesos ``spawn`` de la ejecucion paralela.

Lo que no esta en la tabla cae en ``pipeline.stage_failed`` (o ``evaluation.failed``
cuando la etapa es L3). Las exclusiones de SKU antes de L2 llevan sus propios
codigos ``eligibility.*``.
"""

from __future__ import annotations

from functools import cache
from importlib import import_module
from typing import Final

# Exclusiones de SKU decididas antes de L2 (G7).
ELEGIBILIDAD_SIN_HISTORIA: Final = "eligibility.no_admissible_history"
ELEGIBILIDAD_HUECOS: Final = "eligibility.calendar_gaps"
ELEGIBILIDAD_SIN_RESERVA: Final = "eligibility.no_reserved_observations"
ELEGIBILIDAD_NO_FINITA: Final = "eligibility.non_finite_demand"
ELEGIBILIDAD_HISTORIA_INSUFICIENTE: Final = "eligibility.insufficient_history"

CODIGO_ETAPA_FALLIDA: Final = "pipeline.stage_failed"
CODIGO_EVALUACION_FALLIDA: Final = "evaluation.failed"

# (modulo, clase, codigo). Los hijos van antes que sus bases: se usa ``isinstance``
# en orden de aparicion.
_TABLA: Final[tuple[tuple[str, str, str], ...]] = (
    # L1: ingesta
    ("pred_engine.ingesta.data.deposito", "RawConflictError", "ingest.raw_conflict"),
    (
        "pred_engine.ingesta.data.proteccion_raw",
        "RawWritePermissionError",
        "ingest.raw_write_blocked",
    ),
    (
        "pred_engine.ingesta.lector.extractor_csv",
        "ArchivoNoCsvError",
        "ingest.not_csv",
    ),
    (
        "pred_engine.ingesta.lector.extractor_csv",
        "RutaFueraDeRawError",
        "ingest.not_csv",
    ),
    (
        "pred_engine.ingesta.sonda.errores",
        "SemanticAlignmentError",
        "ingest.header_rejected",
    ),
    (
        "pred_engine.ingesta.validador_formato.errores",
        "SchemaBarrierError",
        "ingest.schema_barrier",
    ),
    (
        "pred_engine.ingesta.continuidad.errores",
        "TemporalContinuityError",
        "ingest.temporal_continuity",
    ),
    (
        "pred_engine.ingesta.categorizacion.errores",
        "TopologyMathError",
        "topology.math",
    ),
    (
        "pred_engine.ingesta.categorizacion.errores",
        "TopologyRoutingError",
        "topology.routing",
    ),
    (
        "pred_engine.ingesta.categorizacion.errores",
        "TopologyContractError",
        "topology.contract",
    ),
    ("pred_engine.ingesta.salida.errores", "OutputContractError", "handoff.contract"),
    (
        "pred_engine.ingesta.salida.errores",
        "PanelPreservationError",
        "handoff.preservation",
    ),
    (
        "pred_engine.ingesta.salida.errores",
        "HandoffPreconditionError",
        "handoff.precondition",
    ),
    (
        "pred_engine.ingesta.salida.errores",
        "MissingCutoffMetadataError",
        "handoff.missing_metadata",
    ),
    ("pred_engine.comun.llm.errores", "LlmError", "ingest.llm_provider"),
    # L2: seleccion y reanudacion
    (
        "pred_engine.optimizacion.router.errores",
        "NingunTrialCompletoError",
        "selection.no_viable_trial",
    ),
    (
        "pred_engine.optimizacion.router.errores",
        "SelectionContractError",
        "selection.contract",
    ),
    (
        "pred_engine.optimizacion.router.errores",
        "UnregisteredFamilyError",
        "selection.unregistered_family",
    ),
    (
        "pred_engine.optimizacion.router.errores",
        "UnknownSkuClassError",
        "selection.unknown_class",
    ),
    (
        "pred_engine.optimizacion.router.errores",
        "RouterConfigurationError",
        "selection.router_config",
    ),
    (
        "pred_engine.optimizacion.router.errores",
        "DuplicateStrategyError",
        "selection.router_config",
    ),
    (
        "pred_engine.optimizacion.control_reanudacion.errores",
        "IncompatibilidadCorridaError",
        "run.incompatible_config",
    ),
    (
        "pred_engine.optimizacion.control_reanudacion.errores",
        "CorridaFallidaError",
        "run.failed_terminal",
    ),
    (
        "pred_engine.optimizacion.control_reanudacion.errores",
        "TransicionEstadoError",
        "run.invalid_transition",
    ),
    (
        "pred_engine.optimizacion.control_reanudacion.errores",
        "ManifiestoAusenteError",
        "run.manifest_missing",
    ),
    (
        "pred_engine.optimizacion.control_reanudacion.errores",
        "ManifiestoCorruptoError",
        "run.manifest_corrupt",
    ),
    (
        "pred_engine.optimizacion.control_reanudacion.errores",
        "VersionManifiestoError",
        "run.manifest_version",
    ),
    # L2/L3: modelos, HPO y evaluacion
    (
        "pred_engine.comun.modelos.modelos_machine_learning.errores",
        "SerieCortaError",
        "model.series_too_short",
    ),
    (
        "pred_engine.comun.walkforward.errores",
        "VentanaInsuficienteError",
        "model.series_too_short",
    ),
    (
        "pred_engine.comun.modelos.modelos_fundacionales.errores",
        "ModeloFundacionalNoDisponibleError",
        "model.foundation_unavailable",
    ),
    (
        "pred_engine.comun.modelos.modelos_clasicos.errores",
        "AjusteModeloError",
        "model.fit_failed",
    ),
    (
        "pred_engine.optimizacion.optimizadores.HPO.errores",
        "EspacioInvalidoError",
        "hpo.invalid_space",
    ),
    (
        "pred_engine.optimizacion.optimizadores.HPO.errores",
        "EstudioError",
        "hpo.study_failed",
    ),
    (
        "pred_engine.comun.walkforward.errores",
        "EvaluacionError",
        CODIGO_EVALUACION_FALLIDA,
    ),
    # Pipeline
    (
        "pred_engine.comun.procedencia",
        "OrigenIncompatibleError",
        "provenance.origin_conflict",
    ),
    ("pred_engine.pipeline", "StageUnavailableError", "pipeline.stage_unavailable"),
    # L0: Fase 0
    (
        "pred_engine.aumentacion.errores",
        "WormOverwriteError",
        "synthetic.worm_overwrite",
    ),
    (
        "pred_engine.aumentacion.errores",
        "SchemaConformanceError",
        "synthetic.conformance",
    ),
    (
        "pred_engine.aumentacion.errores",
        "DivergenceRejectionExhausted",
        "synthetic.divergence_exhausted",
    ),
    (
        "pred_engine.aumentacion.errores",
        "PhysicalConstraintError",
        "synthetic.physical_constraint",
    ),
)

# ``FileNotFoundError`` es de la biblioteca estandar: va al final de la ingesta para
# que ``RawConflictError`` (``FileExistsError``) y los demas tipos propios ganen.
_ARCHIVO_AUSENTE: Final = "ingest.file_not_found"


@cache
def _resueltas() -> tuple[tuple[type[BaseException], str], ...]:
    """Clases de la tabla ya importadas; omite las que no se pueden cargar."""
    resueltas: list[tuple[type[BaseException], str]] = []
    for modulo, nombre, codigo in _TABLA:
        try:
            clase = getattr(import_module(modulo), nombre)
        except (ImportError, AttributeError):  # dependencia opcional ausente
            continue
        resueltas.append((clase, codigo))
    return tuple(resueltas)


def codigos_conocidos() -> frozenset[str]:
    """Todos los codigos que el motor puede emitir (para pruebas y documentacion)."""
    return frozenset(
        {codigo for _, _, codigo in _TABLA}
        | {
            _ARCHIVO_AUSENTE,
            CODIGO_ETAPA_FALLIDA,
            CODIGO_EVALUACION_FALLIDA,
            ELEGIBILIDAD_SIN_HISTORIA,
            ELEGIBILIDAD_HUECOS,
            ELEGIBILIDAD_SIN_RESERVA,
            ELEGIBILIDAD_NO_FINITA,
            ELEGIBILIDAD_HISTORIA_INSUFICIENTE,
        }
    )


def codigo_de(exc: BaseException, *, etapa: str | None = None) -> str:
    """Codigo estable de ``exc``; ``etapa`` solo decide el respaldo de errores sin tipo.

    ``PipelineExecutionError`` envuelve la causa real en ``__cause__``: el codigo es el
    de esa causa, evaluada en la etapa que fallo.
    """
    causa = exc.__cause__
    if type(exc).__name__ == "PipelineExecutionError" and causa is not None:
        return codigo_de(causa, etapa=str(getattr(exc, "stage", etapa)))
    for clase, codigo in _resueltas():
        if isinstance(exc, clase):
            return codigo
    if isinstance(exc, FileNotFoundError):
        return _ARCHIVO_AUSENTE
    if etapa == "L3":
        return CODIGO_EVALUACION_FALLIDA
    return CODIGO_ETAPA_FALLIDA
