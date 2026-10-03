"""Coordinacion de las cuatro etapas principales de PRED en la capa de aplicacion.

No requiere terminal, servidor, motor de workflows ni estado global de corrida.
L2 trabaja solo con la historia admisible (hasta t*, ADR-03-003) y, junto con L3,
ejecuta cada SKU x familia como unidad independiente: en paralelo cuando se
piden varios procesos, y con fallos aislados por unidad (3.0). L3 entrega
evidencia walk-forward, no comparacion estadistica; L4 no tiene implementacion
en el repositorio y la corrida se detiene ahi salvo que el llamador inyecte un
validador real.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import pandas as pd

from pred_engine.comun.dataclasses.validacion_temporal import ResultadoWalkForward
from pred_engine.comun.ejecucion_paralela import (
    RegistroUnidad,
    ResultadoUnidad,
    ejecutar_unidades,
)
from pred_engine.comun.llm import LlmProvider
from pred_engine.comun.modelos import ClassifiedObservation
from pred_engine.comun.reserva import ReserveCut
from pred_engine.comun.walkforward import evaluar_walk_forward
from pred_engine.comun.walkforward.protocolos import (
    FabricaPronosticador,
)
from pred_engine.ingesta.pipeline import (
    run_classify_csv,
    run_ingest,
    run_verify_parquet,
)
from pred_engine.ingesta.salida import validate_output_contract
from pred_engine.optimizacion.optimizadores.HPO.poda import es_degenerada
from pred_engine.optimizacion.router import SelectionRouter
from pred_engine.optimizacion.router.contratos import (
    PredictorFamily,
    RoutingDecision,
    SelectionRequest,
    SelectionResult,
)

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
        "absent",
        "Verdicts and audit bundles are not implemented",
    ),
)


@dataclass(frozen=True, slots=True)
class PipelineInput:
    csv_path: str | Path | None = None
    parquet_path: str | Path | None = None
    data_root: str | Path | None = None
    provider: LlmProvider | None = None
    timeout: float = 30.0

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
    metric: str = "mase"
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


@dataclass(frozen=True, slots=True)
class SkuExclusion:
    """SKU descartado antes de L2 con su causa registrada (3.1)."""

    sku_id: str
    cause: str


@dataclass(frozen=True, slots=True)
class UnitExecution:
    """Traza de una unidad SKU x familia de L2 o L3 (3.0)."""

    stage: StageId
    sku_id: str
    family: PredictorFamily
    record: RegistroUnidad

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


@dataclass(frozen=True, slots=True, eq=False)
class FittingArtifact:
    ingestion: IngestionArtifact
    reserve: ReserveCut
    candidates: tuple[FittedCandidate, ...]
    units: tuple[UnitExecution, ...] = ()
    exclusions: tuple[SkuExclusion, ...] = ()


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
    verdict: Literal["hold", "partial", "fail"]
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sku_id.strip() or self.verdict not in ("hold", "partial", "fail"):
            raise ValueError("A verdict requires a SKU and hold/partial/fail")


@dataclass(frozen=True, slots=True)
class ValidationArtifact:
    verdicts: tuple[SkuVerdict, ...]
    audit_bundle: Mapping[str, Any]


class RetrospectiveValidator(Protocol):
    """Future L4 boundary: consume actual L3 evidence, never a fabricated result."""

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
    if fitting is None:
        return summary
    reserve = fitting.reserve
    evaluation = result.evaluation
    summary["reserve"] = {
        "fraction": reserve.fraction,
        "t_star": reserve.t_star.date().isoformat(),
        "first_reserved": reserve.first_reserved.date().isoformat(),
        "last_observed": reserve.last_observed.date().isoformat(),
        "reserved_days": reserve.reserved_days,
    }
    summary["units"] = {"L2": _unit_counts(fitting.units)}
    if evaluation is not None:
        summary["units"]["L3"] = _unit_counts(evaluation.units)
    summary["exclusions"] = [
        {"sku_id": item.sku_id, "cause": item.cause} for item in fitting.exclusions
    ]
    summary["failures"] = [
        {
            "stage": unit.stage,
            "sku_id": unit.sku_id,
            "family": unit.family,
            "error": unit.record.error,
        }
        for unit in result.failures
    ]
    return summary


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


def _fit_unit(task: _FitTask) -> FittedCandidate:
    """Unidad L2: seleccion de la familia, ajuste final y pronostico desde t*."""
    selection = task.router.execute(task.request, task.decision)
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
            f"desde t*: {motivo} (regla #3)"
        )
    return FittedCandidate(selection, y, forecast)


@dataclass(frozen=True, slots=True, eq=False)
class _EvaluationTask:
    selection: SelectionResult
    series: np.ndarray
    factory: FabricaPronosticador
    settings: EvaluationSettings


def _evaluate_unit(task: _EvaluationTask) -> ResultadoWalkForward:
    """Unidad L3: modelos nuevos por ventana causal, nunca el ajuste final."""
    cfg, selection = task.settings, task.selection
    evidence = evaluar_walk_forward(
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


def _units(
    stage: StageId,
    identities: Sequence[tuple[str, PredictorFamily]],
    results: Sequence[ResultadoUnidad[Any]],
) -> tuple[UnitExecution, ...]:
    return tuple(
        UnitExecution(stage, sku_id, family, result.registro)
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
    ) -> None:
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.router = router
        self.settings = evaluation
        self.factories = dict(factories)
        self.validator = validator
        self.workers = workers

    @property
    def stages(self) -> tuple[StageDefinition, ...]:
        if self.validator is None:
            return PRINCIPAL_STAGES
        return PRINCIPAL_STAGES[:3] + (
            replace(
                PRINCIPAL_STAGES[3],
                maturity="caller_provided",
                capability="Caller-supplied retrospective validator",
            ),
        )

    def ingest(self, request: PipelineInput) -> IngestionArtifact:
        """L1: reuse current ingestion and consume its published 1.4 contract."""
        if request.parquet_path is not None:
            path = Path(request.parquet_path).expanduser().resolve()
        elif request.provider is not None:
            path = run_ingest(
                cast(str | Path, request.csv_path),
                request.provider,
                data_root=request.data_root,
                timeout=request.timeout,
            ).parquet_path
        else:
            _, path = run_classify_csv(
                cast(str | Path, request.csv_path),
                data_root=request.data_root,
            )
        return IngestionArtifact(path, run_verify_parquet(path))

    def _exclusion_cause(
        self, admissible: pd.DataFrame, reserve: ReserveCut
    ) -> str | None:
        if admissible.empty:
            return "sin historia admisible: la serie comienza despues de t*"
        timestamps = admissible["timestamp"]
        if not timestamps.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
            return "calendario diario con huecos (daily grid requerido)"
        if timestamps.iloc[-1] != reserve.t_star:
            return "la serie termina antes de t*: no tiene observaciones reservadas"
        if not np.isfinite(admissible["demand_qty"].to_numpy(dtype=float)).all():
            return "demanda no finita en la historia admisible"
        needed = self.settings.min_train + self.settings.horizon
        if len(admissible) < needed:
            return (
                f"historia admisible insuficiente: {len(admissible)} dias, se "
                f"requieren {needed} (min_train + horizonte)"
            )
        return None

    def fit(self, ingestion: IngestionArtifact) -> FittingArtifact:
        """L2: corte t*, una unidad por SKU x familia y pronostico de la reserva."""
        panel = ingestion.panel
        validate_output_contract(panel)
        reserve = ReserveCut.of(panel)
        tasks: list[_FitTask] = []
        exclusions: list[SkuExclusion] = []
        for sku_id, group in panel.groupby("sku_id", sort=True):
            ordered = group.sort_values("timestamp")
            admissible = ordered.loc[ordered["timestamp"] <= reserve.t_star]
            cause = self._exclusion_cause(admissible, reserve)
            if cause is not None:
                exclusions.append(SkuExclusion(str(sku_id), cause))
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
        results = ejecutar_unidades(_fit_unit, tasks, procesos=self.workers)
        units = _units(
            "L2",
            [(task.request.sku_id, task.decision.family) for task in tasks],
            results,
        )
        candidates = tuple(r.valor for r in results if r.valor is not None)
        if not candidates:
            raise ValueError(_no_results_message("L2", units, exclusions))
        return FittingArtifact(ingestion, reserve, candidates, units, tuple(exclusions))

    def evaluate(self, fitting: FittingArtifact) -> EvaluationArtifact:
        """L3: walk-forward causal por candidato, cada uno como unidad aislada."""
        if not fitting.candidates:
            raise ValueError("L3 requires fitted candidates")
        tasks: list[_EvaluationTask] = []
        for candidate in fitting.candidates:
            family = candidate.selection.family
            factory = self.factories.get(family)
            if factory is None:
                raise ValueError(f"No model factory for {family}")
            tasks.append(
                _EvaluationTask(
                    candidate.selection, candidate.series, factory, self.settings
                )
            )
        results = ejecutar_unidades(_evaluate_unit, tasks, procesos=self.workers)
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
        if not evaluated:
            raise ValueError(_no_results_message("L3", units))
        return EvaluationArtifact(fitting, self.settings, evaluated, units)

    def validate(self, evaluation: EvaluationArtifact) -> ValidationArtifact:
        """L4: an injected validator must cover every SKU; none is built in."""
        if self.validator is None:
            raise StageUnavailableError(PRINCIPAL_STAGES[3].capability)
        result = self.validator(evaluation)
        if not isinstance(result, ValidationArtifact):
            raise TypeError("L4 must return ValidationArtifact")
        expected = {c.fitted.selection.sku_id for c in evaluation.candidates}
        actual = [v.sku_id for v in result.verdicts]
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
        for index, operation in enumerate(operations):
            current = executions[index]
            if current.definition.id == "L4" and self.validator is None:
                executions[index] = replace(
                    current,
                    state="blocked",
                    message=current.definition.capability,
                )
                return PipelineResult(tuple(executions))
            try:
                value = operation(value)
            except Exception as exc:
                executions[index] = replace(current, state="failed", message=str(exc))
                raise PipelineExecutionError(
                    current.definition.id,
                    PipelineResult(tuple(executions)),
                    str(exc),
                ) from exc
            executions[index] = replace(
                current, state="completed", output=value, message=_stage_message(value)
            )
        return PipelineResult(tuple(executions))
