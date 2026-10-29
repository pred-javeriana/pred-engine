"""Coordinacion de las cuatro etapas principales de PRED en la capa de aplicacion.

No requiere terminal, servidor, motor de workflows ni estado global de corrida.
L2 trabaja solo con la historia admisible (hasta t*, ADR-03-003) y, junto con L3,
ejecuta cada SKU x familia como unidad independiente: en paralelo cuando se
piden varios procesos, y con fallos aislados por unidad (3.0). L3 entrega
evidencia walk-forward sobre la historia admisible. L4 integra la evaluacion
reservada de M3 mediante un validador que recibe la conexion de plataforma.
Sin esa configuracion, la biblioteca conserva L4 bloqueada.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import pandas as pd

from pred_engine.comun.dataclasses.validacion_temporal import ResultadoWalkForward
from pred_engine.comun.ejecucion_paralela import (
    EtiquetaUnidad,
    RegistroUnidad,
    ResultadoUnidad,
    ejecutar_unidades,
    emitir_evento,
    enviar_unidad,
    pool_de_unidades,
    resultado_de,
)
from pred_engine.comun.errores_codigo import (
    ELEGIBILIDAD_HISTORIA_INSUFICIENTE,
    ELEGIBILIDAD_HUECOS,
    ELEGIBILIDAD_NO_FINITA,
    ELEGIBILIDAD_SIN_HISTORIA,
    ELEGIBILIDAD_SIN_RESERVA,
    codigo_de,
)
from pred_engine.comun.exclusiones import SkuExclusion
from pred_engine.comun.llm import LlmProvider
from pred_engine.comun.modelos import ClassifiedObservation, TopologyMetrics
from pred_engine.comun.modelos.manifiesto_candidatos import MODELO_POR_FAMILIA
from pred_engine.comun.reserva import ReserveCut
from pred_engine.comun.walkforward import evaluar_walk_forward
from pred_engine.comun.walkforward.protocolos import (
    FabricaPronosticador,
)
from pred_engine.forecasting.evaluaciones.veredictos import VEREDICTOS, Veredicto
from pred_engine.ingesta.categorizacion import classify_panel
from pred_engine.ingesta.informe import ingest_csv_with_report
from pred_engine.ingesta.salida import (
    ClassifiedPublication,
    read_classified_publication,
    validate_output_contract,
)
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda, es_degenerada
from pred_engine.optimizacion.router import SelectionRouter
from pred_engine.optimizacion.router.contratos import (
    PredictorFamily,
    RoutingDecision,
    RoutingPolicy,
    SelectionRequest,
    SelectionResult,
)
from pred_engine.unit_checkpoints import UnitCheckpoints, load_result, save_result

StageId = Literal["L1", "L2", "L3", "L4"]
StageState = Literal["pending", "completed", "blocked", "failed"]


@dataclass(frozen=True, slots=True)
class StageDefinition:
    id: StageId
    name: str
    maturity: Literal["implemented", "partial", "absent", "caller_provided"]
    capability: str


PRINCIPAL_STAGES = (
    StageDefinition(
        "L1",
        "ingestion and characterisation",
        "implemented",
        "Canonical/semantic CSV ingestion and classified Parquet",
    ),
    StageDefinition(
        "L2",
        "model selection and fitting",
        "implemented",
        "20% reserve cut, topology routing, isolated parallel HPO and fitted "
        "forecasts from t*",
    ),
    StageDefinition(
        "L3",
        "walk-forward evaluation",
        "partial",
        "Causal walk-forward metrics; statistical comparison absent",
    ),
    StageDefinition(
        "L4",
        "retrospective validation",
        "implemented",
        "M3 requiere una conexion a la base de plataforma para evaluar y publicar",
    ),
)


@dataclass(frozen=True, slots=True)
class PipelineInput:
    csv_path: str | Path | None = None
    parquet_path: str | Path | None = None
    data_root: str | Path | None = None
    provider: LlmProvider | None = None
    timeout: float = 30.0
    # "keep_both" conserva el CSV nuevo bajo otro nombre si raw/ ya tiene uno
    # distinto con el mismo nombre; "reject" falla sin escribir nada.
    on_conflict: Literal["reject", "keep_both"] = "reject"
    datos_sinteticos: bool | None = None

    def __post_init__(self) -> None:
        if (self.csv_path is None) == (self.parquet_path is None):
            raise ValueError("Provide exactly one CSV or classified Parquet")
        if self.parquet_path is not None and self.provider is not None:
            raise ValueError("Semantic probing applies only to CSV inputs")


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    min_train: int
    horizon: int = 7
    step: int = 7
    metric: str = "rmse"
    seasonality: int = 7
    aggregation: Literal["media", "mediana", "media_recortada"] = "media_recortada"
    trim: float = 0.1

    def __post_init__(self) -> None:
        for name in ("min_train", "horizon", "step", "seasonality"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.metric not in ("mae", "rmse", "smape", "mase"):
            raise ValueError("metric must be mae, rmse, smape or mase")
        if self.metric == "mase" and self.min_train <= self.seasonality:
            raise ValueError("MASE requires min_train greater than seasonality")
        if self.aggregation not in ("media", "mediana", "media_recortada"):
            raise ValueError("Unknown aggregation")
        if not 0 <= self.trim < 0.5:
            raise ValueError("trim must be in [0, 0.5)")


@dataclass(frozen=True, slots=True, eq=False)
class IngestionArtifact:
    parquet_path: Path
    panel: pd.DataFrame
    # Metricas por SKU de la clasificacion (G1) con el corte de la publicacion
    # leida; vacio en un artefacto construido a mano.
    topology: tuple[TopologyMetrics, ...] = ()
    source_sha256: str | None = None
    # Origen declarado en la entrada (CSV resuelto con su informe M1, o Parquet).
    datos_sinteticos: bool | None = None
    # SHA-256 de los bytes Parquet de los que se cargo ``panel``.
    parquet_sha256: str | None = None
    reserve: ReserveCut | None = None
    exclusions: tuple[SkuExclusion, ...] = ()


@dataclass(frozen=True, slots=True)
class UnitExecution:
    """Traza de una unidad SKU x familia de L2 o L3 (3.0)."""

    stage: StageId
    sku_id: str
    family: PredictorFamily
    record: RegistroUnidad
    reused: bool = False

    @property
    def failed(self) -> bool:
        return self.record.estado == "fallida"


@dataclass(frozen=True, slots=True, eq=False)
class FittedCandidate:
    """Configuracion seleccionada, historia admisible y pronostico desde t*.

    ``forecast`` cubre los dias reservados (``ReserveCut.reserved_dates``) con el
    modelo ajustado sobre ``series``; no se compara con la reserva aqui (M3).
    El modelo ajustado no viaja entre procesos: se reconstruye con la fabrica,
    la configuracion y la semilla, que es lo que exige ADR-03-004.
    """

    selection: SelectionResult
    series: np.ndarray
    forecast: np.ndarray
    # Walk-forward que el HPO ya calculo para esta configuracion: el que L3
    # calcularia. No se guarda en el checkpoint de L2 (pesa megas).
    walk_forward: ResultadoWalkForward | None = None


@dataclass(frozen=True, slots=True, eq=False)
class FittingArtifact:
    ingestion: IngestionArtifact
    reserve: ReserveCut
    candidates: tuple[FittedCandidate, ...]
    units: tuple[UnitExecution, ...] = ()
    exclusions: tuple[SkuExclusion, ...] = ()
    # Politica con la que se enruto: dice que familias corresponden a cada clase.
    policy: RoutingPolicy | None = None
    # Ajustes de M2 que cambian resultados (familias, ensayos, semilla).
    m2_settings: Mapping[str, Any] = field(default_factory=dict)
    # Exclusiones de M1 (lanzados despues del corte) que L2 leyo con ``reserve``;
    # M3 las publica tal cual, sin recalcularlas.
    late_launches: tuple[SkuExclusion, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class EvaluatedCandidate:
    fitted: FittedCandidate
    walk_forward: ResultadoWalkForward


@dataclass(frozen=True, slots=True, eq=False)
class EvaluationArtifact:
    fitting: FittingArtifact
    settings: EvaluationSettings
    candidates: tuple[EvaluatedCandidate, ...]
    units: tuple[UnitExecution, ...] = ()
    # Configuration search and evaluation use the same history. This is not an
    # independent holdout verdict, even though each window refit is causal.
    selection_scope: str = "same_history"
    statistical_comparison_available: bool = False


@dataclass(frozen=True, slots=True)
class SkuVerdict:
    sku_id: str
    verdict: Veredicto | Literal["hold", "partial", "fail"]
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sku_id.strip() or self.verdict not in (
            *VEREDICTOS,
            "hold",
            "partial",
            "fail",
        ):
            raise ValueError("Un veredicto requiere SKU y una etiqueta reconocida")


@dataclass(frozen=True, slots=True)
class ValidationArtifact:
    verdicts: tuple[SkuVerdict, ...]
    audit_bundle: Mapping[str, Any]


class RetrospectiveValidator(Protocol):
    """L4 boundary for native M3 or a caller-provided retrospective validator."""

    def __call__(self, evaluation: EvaluationArtifact) -> ValidationArtifact: ...


PipelineArtifact = (
    IngestionArtifact | FittingArtifact | EvaluationArtifact | ValidationArtifact
)


@dataclass(frozen=True, slots=True, eq=False)
class StageExecution:
    definition: StageDefinition
    state: StageState = "pending"
    output: PipelineArtifact | None = None
    message: str = ""
    # Reloj de pared (epoch, segundos) de la etapa; None si no llego a correr.
    started_at: float | None = None
    finished_at: float | None = None


@dataclass(frozen=True, slots=True, eq=False)
class PipelineResult:
    stages: tuple[StageExecution, ...]

    @property
    def complete(self) -> bool:
        return len(self.stages) == 4 and all(
            stage.state == "completed" for stage in self.stages
        )

    @property
    def blocked_stage(self) -> StageId | None:
        return next(
            (stage.definition.id for stage in self.stages if stage.state == "blocked"),
            None,
        )

    @property
    def ingestion(self) -> IngestionArtifact | None:
        return cast(IngestionArtifact | None, self.stages[0].output)

    @property
    def fitting(self) -> FittingArtifact | None:
        return cast(FittingArtifact | None, self.stages[1].output)

    @property
    def evaluation(self) -> EvaluationArtifact | None:
        return cast(EvaluationArtifact | None, self.stages[2].output)

    @property
    def validation(self) -> ValidationArtifact | None:
        return cast(ValidationArtifact | None, self.stages[3].output)

    @property
    def units(self) -> tuple[UnitExecution, ...]:
        fitting, evaluation = self.fitting, self.evaluation
        return (fitting.units if fitting else ()) + (
            evaluation.units if evaluation else ()
        )

    @property
    def failures(self) -> tuple[UnitExecution, ...]:
        """Unidades aisladas que fallaron sin detener su etapa."""
        return tuple(unit for unit in self.units if unit.failed)


def _unit_counts(units: Sequence[UnitExecution]) -> dict[str, int]:
    failed = sum(unit.failed for unit in units)
    return {"total": len(units), "completed": len(units) - failed, "failed": failed}


def summarize_run(result: PipelineResult) -> dict[str, Any]:
    """Vista diagnostica JSON y determinista; no serializa modelos ni demanda."""
    fitting = result.fitting
    summary: dict[str, Any] = {
        "complete": result.complete,
        "blocked_stage": result.blocked_stage,
        "stages": [
            {
                "id": stage.definition.id,
                "name": stage.definition.name,
                "maturity": stage.definition.maturity,
                "state": stage.state,
                "capability": stage.definition.capability,
                "message": stage.message,
            }
            for stage in result.stages
        ],
    }
    if result.validation is not None:
        summary["m3"] = dict(result.validation.audit_bundle)
    ingestion = result.ingestion
    # El corte ya es conocido desde L1: una corrida que falla en L2 lo conserva.
    reserve = (
        fitting.reserve
        if fitting is not None
        else (ingestion.reserve if ingestion is not None else None)
    )
    if reserve is not None:
        summary["reserve"] = {
            "fraction": reserve.fraction,
            "t_star": reserve.t_star.date().isoformat(),
            "first_reserved": reserve.first_reserved.date().isoformat(),
            "last_observed": reserve.last_observed.date().isoformat(),
            "reserved_days": reserve.reserved_days,
        }
    if fitting is None:
        if ingestion is not None and ingestion.exclusions:
            summary["excluded_count"] = len(ingestion.exclusions)
            summary["exclusions"] = [
                {"sku_id": item.sku_id, "cause": item.cause, "code": item.code}
                for item in ingestion.exclusions
            ]
        else:
            summary["excluded_count"] = 0
            summary["exclusions"] = []
        return summary
    evaluation = result.evaluation
    summary["units"] = {"L2": _unit_counts(fitting.units)}
    if evaluation is not None:
        summary["units"]["L3"] = _unit_counts(evaluation.units)
    summary["excluded_count"] = len(fitting.exclusions)
    summary["exclusions"] = [
        {"sku_id": item.sku_id, "cause": item.cause, "code": item.code}
        for item in fitting.exclusions
    ]
    summary["failures"] = [
        {
            "stage": unit.stage,
            "sku_id": unit.sku_id,
            "family": unit.family,
            "error": unit.record.error,
            "code": unit.record.codigo,
        }
        for unit in result.failures
    ]
    return summary


def ingestion_artifact_from_publication(
    path: str | Path,
    publication: ClassifiedPublication,
    *,
    source_sha256: str | None = None,
    datos_sinteticos: bool | None = None,
) -> IngestionArtifact:
    """Artefacto L1 cuyas filas, huella, corte, exclusiones y topologia salen
    de una sola lectura de la publicacion (``read_classified_publication``).

    Es la ruta de biblioteca para armar la entrada de ``Pipeline.fit`` sin
    ``Pipeline.ingest``; nunca vuelve a leer ``path``.
    """
    meta = publication.metadata
    topologia = classify_panel(publication.frame, corte=meta.reserve)
    return IngestionArtifact(
        Path(path),
        publication.frame,
        topology=tuple(topologia.metrics),
        source_sha256=source_sha256,
        datos_sinteticos=datos_sinteticos,
        parquet_sha256=publication.sha256,
        reserve=meta.reserve,
        exclusions=meta.exclusions,
    )


class StageUnavailableError(RuntimeError):
    pass


class PipelineExecutionError(RuntimeError):
    """Retains completed evidence and the failed stage; original error is __cause__."""

    def __init__(self, stage: StageId, result: PipelineResult, message: str) -> None:
        self.stage = stage
        self.result = result
        super().__init__(f"{stage}: {message}")


@dataclass(frozen=True, slots=True, eq=False)
class _FitTask:
    router: SelectionRouter
    request: SelectionRequest
    decision: RoutingDecision
    factory: FabricaPronosticador
    horizon: int
    checkpoint: Path | None = None


def _fit_unit(task: _FitTask) -> FittedCandidate:
    """Unidad L2: seleccion de la familia, ajuste final y pronostico desde t*."""
    # Un router que solo sabe `execute` (un envoltorio, por ejemplo) no da walk-forward.
    ejecutar = getattr(task.router, "execute_with_walk_forward", None)
    if ejecutar is None:
        selection, walk_forward = task.router.execute(task.request, task.decision), None
    else:
        selection, walk_forward = ejecutar(task.request, task.decision)
    if selection.forecast_config is None:
        raise ValueError(f"{selection.family} must provide forecast_config for L2")
    y = np.array([obs.demand_qty for obs in task.request.series], dtype=float)
    model = task.factory(selection.forecast_config, seed=selection.forecast_seed)
    forecast = np.asarray(model.fit(y.copy()).predict(task.horizon), dtype=float)
    if forecast.shape != (task.horizon,):
        raise ValueError(
            f"pronostico invalido para {selection.sku_id}/{selection.family}: "
            f"se esperaban {task.horizon} valores"
        )
    # Regla #3 (ADR-020): el pronostico que recibe M3 cumple la misma regla que
    # cada ventana del HPO; uno degenerado hace fallar la unidad, no se entrega.
    degenerado, motivo = es_degenerada(forecast, y_train=y)
    if degenerado:
        raise ValueError(
            f"pronostico degenerado para {selection.sku_id}/{selection.family} "
            f"desde t*: {motivo} (regla #3); "
            f"configuracion={dict(selection.forecast_config)}"
        )
    return FittedCandidate(selection, y, forecast, walk_forward)


@dataclass(frozen=True, slots=True, eq=False)
class _EvaluationTask:
    selection: SelectionResult
    series: np.ndarray
    factory: FabricaPronosticador
    settings: EvaluationSettings
    checkpoint: Path | None = None
    walk_forward: ResultadoWalkForward | None = None


def _evaluate_unit(task: _EvaluationTask) -> ResultadoWalkForward:
    """Unidad L3: modelos nuevos por ventana causal, nunca el ajuste final."""
    cfg, selection = task.settings, task.selection
    # El HPO ya evaluo esta configuracion en las mismas ventanas: no se repite.
    evidence = task.walk_forward or evaluar_walk_forward(
        task.series,
        task.factory,
        cast(Mapping[str, Any], selection.forecast_config),
        min_train=cfg.min_train,
        horizonte=cfg.horizon,
        paso=cfg.step,
        metrica_objetivo=cfg.metric,
        estacionalidad=cfg.seasonality,
        agregacion=cfg.aggregation,
        proporcion_recorte=cfg.trim,
        seed=selection.forecast_seed,
        tolerar_fallos=False,
        identificador=selection.sku_id,
    )
    if not evidence.completo or not np.isfinite(evidence.valor_agregado):
        raise ValueError(
            f"No complete finite walk-forward evidence for {selection.sku_id}"
        )
    return evidence


def _persist_fit(task: _FitTask, result: ResultadoUnidad[FittedCandidate]) -> None:
    # El walk-forward del HPO no se guarda: L3 lo recalcula si el L2 se restaura.
    if result.valor is not None and result.valor.walk_forward is not None:
        result = replace(result, valor=replace(result.valor, walk_forward=None))
    save_result(task.checkpoint, result)


def _persist_evaluation(
    task: _EvaluationTask, result: ResultadoUnidad[ResultadoWalkForward]
) -> None:
    save_result(task.checkpoint, result)


def _units(
    stage: StageId,
    identities: Sequence[tuple[str, PredictorFamily]],
    results: Sequence[ResultadoUnidad[Any]],
) -> tuple[UnitExecution, ...]:
    return tuple(
        UnitExecution(stage, sku_id, family, result.registro, reused=result.reutilizado)
        for (sku_id, family), result in zip(identities, results, strict=True)
    )


def _no_results_message(
    stage: StageId,
    units: Sequence[UnitExecution],
    exclusions: Sequence[SkuExclusion] = (),
) -> str:
    causes = [
        f"{unit.sku_id}/{unit.family}: {unit.record.error}"
        for unit in units
        if unit.failed
    ] + [f"{item.sku_id}: {item.cause}" for item in exclusions]
    return (
        f"{stage} sin resultados ({len(units)} unidades, {len(exclusions)} SKU "
        "excluidos): " + "; ".join(causes[:5])
    )


def _stage_message(output: PipelineArtifact) -> str:
    units = getattr(output, "units", ())
    exclusions = getattr(output, "exclusions", ())
    failed = sum(unit.failed for unit in units)
    parts = []
    if failed:
        parts.append(f"{failed} de {len(units)} unidades fallaron (aisladas)")
    if exclusions:
        parts.append(f"{len(exclusions)} SKU excluidos con causa")
    return "; ".join(parts)


class Pipeline:
    """Fixed four-stage application service with separately callable stage methods.

    ``workers`` es la cantidad de procesos para las unidades de L2 y L3; con 1
    todo corre en el proceso actual.
    """

    def __init__(
        self,
        router: SelectionRouter,
        evaluation: EvaluationSettings,
        *,
        factories: Mapping[PredictorFamily, FabricaPronosticador],
        validator: RetrospectiveValidator | None = None,
        workers: int = 1,
        progress: str | Path | None = None,
        m2_settings: Mapping[str, Any] | None = None,
        checkpoints: UnitCheckpoints | None = None,
    ) -> None:
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.router = router
        # Result-changing M2 settings the router was built with; the native
        # validator records them, so a composition must declare them.
        self.m2_settings = dict(m2_settings or {})
        self.settings = evaluation
        self.factories = dict(factories)
        self.validator = validator
        self.workers = workers
        # Ruta del flujo de progreso (``progreso.jsonl``); None lo desactiva.
        self.progress = progress
        self.checkpoints = checkpoints
        # Pool que comparten L2 y L3 mientras corre ``run`` (con mas de un proceso).
        self._pool: ProcessPoolExecutor | None = None
        # L3 enviadas apenas termino su L2, por (sku, familia), con su hora de envio.
        self._early_l3: dict[
            tuple[str, str], tuple[Future[ResultadoUnidad[ResultadoWalkForward]], float]
        ] = {}

    def _run_units[T, R](
        self,
        function: Callable[[T], R],
        tasks: Sequence[T],
        labels: Sequence[EtiquetaUnidad],
        paths: Sequence[Path | None],
        persist: Callable[[T, ResultadoUnidad[R]], None],
        *,
        submitted: Mapping[int, tuple[Future[ResultadoUnidad[R]], float]] | None = None,
        on_result: Callable[[ResultadoUnidad[R]], None] | None = None,
    ) -> list[ResultadoUnidad[R]]:
        """``submitted``: unidades ya enviadas al pool, por indice. Con el pool,
        ``on_result`` recibe cada resultado (restaurado o nuevo) apenas existe."""
        submitted = submitted or {}
        # Una unidad ya enviada guarda su checkpoint al terminar: no es restaurada.
        restored = [
            None if i in submitted else load_result(path)
            for i, path in enumerate(paths)
        ]
        pending = [i for i, result in enumerate(restored) if result is None]
        for label, result in zip(labels, restored, strict=True):
            if result is not None:
                record = result.registro
                self._emit(
                    "unit_finished",
                    **label.como_dict(),
                    state="completada",
                    pid=record.pid,
                    started_at=datetime.fromtimestamp(record.inicio, UTC).isoformat(),
                    finished_at=datetime.fromtimestamp(record.fin, UTC).isoformat(),
                    seconds=round(record.duracion_s, 3),
                    cpu_s=record.cpu_s,
                    error=None,
                    code=None,
                    reused=True,
                )
        if self._pool is None:
            fresh = ejecutar_unidades(
                function,
                [tasks[i] for i in pending],
                procesos=self.workers,
                etiquetas=[labels[i] for i in pending],
                progreso=self.progress,
                al_completar=persist,
            )
        else:
            sent = {
                i: submitted.get(i)
                or (
                    enviar_unidad(
                        self._pool,
                        function,
                        tasks[i],
                        labels[i],
                        self.progress,
                        labels[i].etapa,
                        persist,
                    ),
                    time.time(),
                )
                for i in pending
            }
            if on_result is not None:
                for result in restored:
                    if result is not None:
                        on_result(result)
            # Cada resultado se atiende al terminar, no en el orden de las tareas.
            done_by_index: dict[int, ResultadoUnidad[R]] = {}
            waiting = {future: i for i, (future, _) in sent.items()}
            while waiting:
                done, _ = wait(waiting, return_when=FIRST_COMPLETED)
                for future in done:
                    i = waiting.pop(future)
                    result = resultado_de(future, labels[i], self.progress, sent[i][1])
                    done_by_index[i] = result
                    if on_result is not None:
                        on_result(result)
            fresh = [done_by_index[i] for i in pending]
        for index, result in zip(pending, fresh, strict=True):
            restored[index] = result
        return cast(list[ResultadoUnidad[R]], restored)

    @contextmanager
    def _pool_l2_l3(self) -> Iterator[None]:
        """Un solo pool para las unidades de L2 y L3 cuando hay mas de un proceso."""
        if self.workers == 1:
            yield
            return
        with pool_de_unidades(self.workers) as pool:
            self._pool = pool
            try:
                yield
            finally:
                self._pool = None
                self._early_l3.clear()

    def _evaluation_task(self, candidate: FittedCandidate) -> _EvaluationTask:
        family = candidate.selection.family
        factory = self.factories.get(family)
        if factory is None:
            raise ValueError(f"No model factory for {family}")
        cfg = self.settings
        ajustes_l3 = {
            "min_train": cfg.min_train,
            "horizonte": cfg.horizon,
            "paso": cfg.step,
            "estacionalidad": cfg.seasonality,
            "metrica_objetivo": cfg.metric,
            "agregacion": cfg.aggregation,
            "proporcion_recorte": cfg.trim,
        }
        # El walk-forward del HPO solo vale si se calculo con los ajustes de L3.
        walk_forward = candidate.walk_forward
        if walk_forward is not None and walk_forward.ajustes != ajustes_l3:
            walk_forward = None
        task = _EvaluationTask(
            candidate.selection,
            candidate.series,
            factory,
            cfg,
            walk_forward=walk_forward,
        )
        if self.checkpoints is None:
            return task
        return replace(
            task,
            checkpoint=self.checkpoints.path(
                "L3",
                {
                    "selection": task.selection.model_dump(mode="json"),
                    "series": task.series.tolist(),
                    "evaluation": asdict(task.settings),
                },
            ),
        )

    def _submit_early_l3(self, result: ResultadoUnidad[FittedCandidate]) -> None:
        """Envia el L3 de un candidato apenas termina su L2, si no esta restaurado."""
        candidate = result.valor
        if candidate is None or self._pool is None:
            return
        task = self._evaluation_task(candidate)
        if load_result(task.checkpoint) is not None:
            return
        sku_id, family = candidate.selection.sku_id, candidate.selection.family
        self._early_l3[(sku_id, family)] = (
            enviar_unidad(
                self._pool,
                _evaluate_unit,
                task,
                self._label("L3", sku_id, family),
                self.progress,
                "L3",
                _persist_evaluation,
            ),
            time.time(),
        )

    def _emit(self, event: str, /, **data: Any) -> None:
        if self.progress is not None:
            emitir_evento(self.progress, event, **data)

    @staticmethod
    def _label(stage: str, sku_id: str, family: PredictorFamily) -> EtiquetaUnidad:
        return EtiquetaUnidad(stage, sku_id, family, MODELO_POR_FAMILIA[family])

    def _plan(self, stage: str, units: list[EtiquetaUnidad]) -> None:
        self._emit(
            "planned",
            stage=stage,
            units=[
                {"sku_id": u.sku_id, "family": u.family, "model": u.model}
                for u in units
            ],
        )

    @property
    def stages(self) -> tuple[StageDefinition, ...]:
        if self.validator is None:
            return PRINCIPAL_STAGES
        return PRINCIPAL_STAGES[:3] + (
            replace(
                PRINCIPAL_STAGES[3],
                maturity="implemented"
                if getattr(self.validator, "nativo", False)
                else "caller_provided",
                capability="Evaluacion retrospectiva nativa y publicacion de evidencia"
                if getattr(self.validator, "nativo", False)
                else "Caller-supplied retrospective validator",
            ),
        )

    def ingest(self, request: PipelineInput) -> IngestionArtifact:
        """L1: reuse current ingestion and consume its published 1.4 contract."""
        if request.parquet_path is not None:
            path = Path(request.parquet_path).expanduser().resolve()
            return ingestion_artifact_from_publication(
                path,
                read_classified_publication(path),
                datos_sinteticos=request.datos_sinteticos,
            )
        salida = ingest_csv_with_report(
            cast(str | Path, request.csv_path),
            data_root=request.data_root,
            provider=request.provider,
            timeout=request.timeout,
            on_conflict=request.on_conflict,
            datos_sinteticos=request.datos_sinteticos,
        )
        if salida.error is not None:
            raise salida.error
        assert salida.result is not None
        # Se usa la publicacion releida, no el artefacto de la importacion en
        # memoria: otra publicacion pudo sustituir el archivo entre ambos pasos.
        path = salida.result.parquet_path
        return ingestion_artifact_from_publication(
            path,
            read_classified_publication(path),
            source_sha256=salida.report.ingest_id,
            datos_sinteticos=salida.report.source_file.datos_sinteticos,
        )

    def _exclusion_cause(
        self, admissible: pd.DataFrame, reserve: ReserveCut
    ) -> tuple[str, str] | None:
        """``(codigo, causa)`` de la exclusion, o ``None`` si el SKU es elegible."""
        if admissible.empty:
            return (
                ELEGIBILIDAD_SIN_HISTORIA,
                "sin historia admisible: la serie comienza despues de t*",
            )
        timestamps = admissible["timestamp"]
        if not timestamps.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
            return (
                ELEGIBILIDAD_HUECOS,
                "calendario diario con huecos (daily grid requerido)",
            )
        if timestamps.iloc[-1] != reserve.t_star:
            return (
                ELEGIBILIDAD_SIN_RESERVA,
                "la serie termina antes de t*: no tiene observaciones reservadas",
            )
        if not np.isfinite(admissible["demand_qty"].to_numpy(dtype=float)).all():
            return (
                ELEGIBILIDAD_NO_FINITA,
                "demanda no finita en la historia admisible",
            )
        pasos = ReglasPoda().min_ventanas - 1
        needed = (
            self.settings.min_train + self.settings.horizon + pasos * self.settings.step
        )
        if len(admissible) < needed:
            return (
                ELEGIBILIDAD_HISTORIA_INSUFICIENTE,
                f"historia admisible insuficiente: {len(admissible)} dias, se "
                f"requieren {needed} (min_train + horizonte + {pasos} pasos)",
            )
        return None

    def fit(self, ingestion: IngestionArtifact) -> FittingArtifact:
        """L2: corte t*, una unidad por SKU x familia y pronostico de la reserva."""
        panel = ingestion.panel
        validate_output_contract(panel)
        reserve = ingestion.reserve
        if reserve is None:
            # La ruta pudo cambiar de contenido despues de leer las filas: el
            # corte nunca se relee de ella.
            raise ValueError(
                "la ingestion no declara su corte: use Pipeline.ingest o "
                "ingestion_artifact_from_publication con read_classified_publication"
            )
        exclusions_l1 = ingestion.exclusions
        tasks: list[_FitTask] = []
        exclusions: list[SkuExclusion] = [
            SkuExclusion(
                e.sku_id,
                f"sin historia admisible: {e.cause}",
                code=ELEGIBILIDAD_SIN_HISTORIA,
            )
            for e in exclusions_l1
        ]
        for item in exclusions:
            self._emit("excluded", sku_id=item.sku_id, cause=item.cause, code=item.code)
        for sku_id, group in panel.groupby("sku_id", sort=True):
            ordered = group.sort_values("timestamp")
            admissible = ordered.loc[ordered["timestamp"] <= reserve.t_star]
            exclusion = self._exclusion_cause(admissible, reserve)
            if exclusion is not None:
                code, cause = exclusion
                exclusions.append(SkuExclusion(str(sku_id), cause, code))
                self._emit("excluded", sku_id=str(sku_id), cause=cause, code=code)
                continue
            series = tuple(
                ClassifiedObservation(**row)
                for row in admissible.to_dict(orient="records")
            )
            request = SelectionRequest(
                sku_id=series[0].sku_id,
                sku_class=series[0].sku_class,
                series=series,
            )
            for decision in self.router.plan(request):
                factory = self.factories.get(decision.family)
                if factory is None:
                    raise ValueError(f"No model factory for {decision.family}")
                tasks.append(
                    _FitTask(
                        self.router, request, decision, factory, reserve.reserved_days
                    )
                )
        labels = [
            self._label("L2", task.request.sku_id, task.decision.family)
            for task in tasks
        ]
        if self.checkpoints is not None:
            tasks = [
                replace(
                    task,
                    checkpoint=self.checkpoints.path(
                        "L2",
                        {
                            "request": task.request.model_dump(mode="json"),
                            "decision": task.decision.model_dump(mode="json"),
                            "horizon": task.horizon,
                            "evaluation": asdict(self.settings),
                            "m2": self.m2_settings,
                        },
                    ),
                )
                for task in tasks
            ]
        self._plan("L2", labels)
        results = self._run_units(
            _fit_unit,
            tasks,
            labels,
            [t.checkpoint for t in tasks],
            _persist_fit,
            on_result=self._submit_early_l3,
        )
        units = _units(
            "L2",
            [(task.request.sku_id, task.decision.family) for task in tasks],
            results,
        )
        candidates = tuple(r.valor for r in results if r.valor is not None)
        if not candidates and not (
            getattr(self.validator, "nativo", False) and any(u.failed for u in units)
        ):
            raise ValueError(_no_results_message("L2", units, exclusions))
        return FittingArtifact(
            ingestion,
            reserve,
            candidates,
            units,
            tuple(exclusions),
            getattr(self.router, "policy", None),
            dict(self.m2_settings),
            tuple(exclusions_l1),
        )

    def evaluate(self, fitting: FittingArtifact) -> EvaluationArtifact:
        """L3: walk-forward causal por candidato, cada uno como unidad aislada."""
        if not fitting.candidates and not (
            getattr(self.validator, "nativo", False)
            and any(u.failed for u in fitting.units)
        ):
            raise ValueError("L3 requires fitted candidates")
        tasks = [self._evaluation_task(c) for c in fitting.candidates]
        labels = [
            self._label("L3", c.selection.sku_id, c.selection.family)
            for c in fitting.candidates
        ]
        self._plan("L3", labels)
        early = {
            i: self._early_l3.pop(key)
            for i, c in enumerate(fitting.candidates)
            if (key := (c.selection.sku_id, c.selection.family)) in self._early_l3
        }
        results = self._run_units(
            _evaluate_unit,
            tasks,
            labels,
            [t.checkpoint for t in tasks],
            _persist_evaluation,
            submitted=early,
        )
        units = _units(
            "L3",
            [(c.selection.sku_id, c.selection.family) for c in fitting.candidates],
            results,
        )
        evaluated = tuple(
            EvaluatedCandidate(candidate, result.valor)
            for candidate, result in zip(fitting.candidates, results, strict=True)
            if result.valor is not None
        )
        if not evaluated and fitting.candidates:
            raise ValueError(_no_results_message("L3", units))
        return EvaluationArtifact(fitting, self.settings, evaluated, units)

    def validate(self, evaluation: EvaluationArtifact) -> ValidationArtifact:
        """L4: native covers handoff SKUs; legacy covers evaluated SKUs.

        A reused native publication covers its frozen handoff, not the SKUs a
        later M2 retry happened to fit.
        """
        if self.validator is None:
            raise StageUnavailableError(PRINCIPAL_STAGES[3].capability)
        result = self.validator(evaluation)
        if not isinstance(result, ValidationArtifact):
            raise TypeError("L4 must return ValidationArtifact")
        actual = [v.sku_id for v in result.verdicts]
        nativo = getattr(self.validator, "nativo", False)
        expected = (
            set(actual)
            if nativo and result.audit_bundle.get("publicada_previamente")
            else (
                {c.selection.sku_id for c in evaluation.fitting.candidates}
                | {u.sku_id for u in evaluation.fitting.units if u.failed}
            )
            if nativo
            else {c.fitted.selection.sku_id for c in evaluation.candidates}
        )
        if not expected or set(actual) != expected or len(actual) != len(expected):
            raise ValueError("L4 must return exactly one verdict for every SKU")
        if not result.audit_bundle:
            raise ValueError("L4 must return audit evidence")
        return result

    def run(self, request: PipelineInput) -> PipelineResult:
        """The only transition loop, shared by library callers, CLI and verification."""
        executions = [StageExecution(stage) for stage in self.stages]
        operations: tuple[Callable[[Any], PipelineArtifact], ...] = (
            self.ingest,
            self.fit,
            self.evaluate,
            self.validate,
        )
        value: Any = request
        # L2 y L3 comparten un pool, que se cierra antes de M3 o si la corrida falla.
        with ExitStack() as pool:
            for index, operation in enumerate(operations):
                current = executions[index]
                if current.definition.id == "L2":
                    pool.enter_context(self._pool_l2_l3())
                elif current.definition.id == "L4":
                    pool.close()
                if current.definition.id == "L4" and self.validator is None:
                    executions[index] = replace(
                        current,
                        state="blocked",
                        message=current.definition.capability,
                    )
                    return PipelineResult(tuple(executions))
                current = replace(current, started_at=time.time())
                self._emit("stage_started", stage=current.definition.id)
                try:
                    value = operation(value)
                except Exception as exc:
                    executions[index] = replace(
                        current,
                        state="failed",
                        message=str(exc),
                        finished_at=time.time(),
                    )
                    self._emit(
                        "stage_finished",
                        stage=current.definition.id,
                        state="failed",
                        message=str(exc),
                        code=codigo_de(exc, etapa=str(current.definition.id)),
                    )
                    raise PipelineExecutionError(
                        current.definition.id,
                        PipelineResult(tuple(executions)),
                        str(exc),
                    ) from exc
                executions[index] = replace(
                    current,
                    state="completed",
                    output=value,
                    message=_stage_message(value),
                    finished_at=time.time(),
                )
                self._emit(
                    "stage_finished", stage=current.definition.id, state="completed"
                )
            return PipelineResult(tuple(executions))
