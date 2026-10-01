"""Application-layer coordination of PRED's four principal stages.

No terminal, server, workflow engine or global run state is required. L3 currently
provides walk-forward evidence, not statistical comparison; L4 has no repository
implementation. A run stops there unless a real validator is supplied by a caller.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import pandas as pd

from pred_engine.comun.dataclasses.validacion_temporal import ResultadoWalkForward
from pred_engine.comun.llm import LlmProvider
from pred_engine.comun.modelos import ClassifiedObservation
from pred_engine.comun.walkforward import evaluar_walk_forward
from pred_engine.comun.walkforward.protocolos import (
    FabricaPronosticador,
    Pronosticador,
)
from pred_engine.ingesta.pipeline import (
    run_classify_csv,
    run_ingest,
    run_verify_parquet,
)
from pred_engine.ingesta.salida import validate_output_contract
from pred_engine.optimizacion.router import SelectionRouter
from pred_engine.optimizacion.router.contratos import (
    PredictorFamily,
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
        "Topology routing, HPO and fitted model instances",
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


@dataclass(frozen=True, slots=True, eq=False)
class FittedCandidate:
    selection: SelectionResult
    series: np.ndarray
    model: Pronosticador


@dataclass(frozen=True, slots=True, eq=False)
class FittingArtifact:
    ingestion: IngestionArtifact
    candidates: tuple[FittedCandidate, ...]


@dataclass(frozen=True, slots=True, eq=False)
class EvaluatedCandidate:
    fitted: FittedCandidate
    walk_forward: ResultadoWalkForward


@dataclass(frozen=True, slots=True, eq=False)
class EvaluationArtifact:
    fitting: FittingArtifact
    settings: EvaluationSettings
    candidates: tuple[EvaluatedCandidate, ...]
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


def summarize_run(result: PipelineResult) -> dict[str, Any]:
    """Small JSON-safe diagnostic view; does not serialize models or raw demand."""
    return {
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


class StageUnavailableError(RuntimeError):
    pass


class PipelineExecutionError(RuntimeError):
    """Retains completed evidence and the failed stage; original error is __cause__."""

    def __init__(self, stage: StageId, result: PipelineResult, message: str) -> None:
        self.stage = stage
        self.result = result
        super().__init__(f"{stage}: {message}")


class Pipeline:
    """Fixed four-stage application service with separately callable stage methods."""

    def __init__(
        self,
        router: SelectionRouter,
        evaluation: EvaluationSettings,
        *,
        factories: Mapping[PredictorFamily, FabricaPronosticador],
        validator: RetrospectiveValidator | None = None,
    ) -> None:
        self.router = router
        self.settings = evaluation
        self.factories = dict(factories)
        self.validator = validator

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

    def fit(self, ingestion: IngestionArtifact) -> FittingArtifact:
        """L2: each SKU becomes a typed request, then selected models are fitted."""
        panel = ingestion.panel
        validate_output_contract(panel)
        candidates: list[FittedCandidate] = []
        for _, group in panel.groupby("sku_id", sort=True):
            ordered = group.sort_values("timestamp")
            timestamps = ordered["timestamp"]
            if not timestamps.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
                raise ValueError("L2 requires a daily grid for every SKU")
            series = tuple(
                ClassifiedObservation(**row)
                for row in ordered.to_dict(orient="records")
            )
            request = SelectionRequest(
                sku_id=series[0].sku_id,
                sku_class=series[0].sku_class,
                series=series,
            )
            y = ordered["demand_qty"].to_numpy(dtype=float, copy=True)
            if not np.isfinite(y).all():
                raise ValueError(f"Non-finite demand for SKU {request.sku_id!r}")
            for selection in self.router.route(request):
                if selection.forecast_config is None:
                    raise ValueError(
                        f"{selection.family} must provide forecast_config for L2"
                    )
                factory = self.factories.get(selection.family)
                if factory is None:
                    raise ValueError(f"No model factory for {selection.family}")
                model = factory(
                    selection.forecast_config,
                    seed=selection.forecast_seed,
                ).fit(y.copy())
                candidates.append(FittedCandidate(selection, y.copy(), model))
        return FittingArtifact(ingestion, tuple(candidates))

    def evaluate(self, fitting: FittingArtifact) -> EvaluationArtifact:
        """L3: fresh models per causal window, never the full-history fitted model."""
        if not fitting.candidates:
            raise ValueError("L3 requires fitted candidates")
        cfg = self.settings
        evaluated: list[EvaluatedCandidate] = []
        for candidate in fitting.candidates:
            selection = candidate.selection
            evidence = evaluar_walk_forward(
                candidate.series,
                self.factories[selection.family],
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
            evaluated.append(EvaluatedCandidate(candidate, evidence))
        return EvaluationArtifact(fitting, cfg, tuple(evaluated))

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
            executions[index] = replace(current, state="completed", output=value)
        return PipelineResult(tuple(executions))
