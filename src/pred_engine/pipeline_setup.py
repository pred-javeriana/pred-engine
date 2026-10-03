"""Repository-native composition, separate from execution and presentation.

Applications may instead inject their existing router, model factories and L4
validator into Pipeline. The initial topology matrix is never changed here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pred_engine.comun.modelos import SkuClass
from pred_engine.comun.modelos.modelos_deep_learning import fabrica_dl
from pred_engine.comun.modelos.modelos_fundacionales.chronos2 import fabrica_fundacional
from pred_engine.comun.modelos.modelos_machine_learning.lgbm import fabrica_ml
from pred_engine.comun.walkforward.protocolos import FabricaPronosticador
from pred_engine.optimizacion.optimizadores.modelos_clasicos import fabrica_sarima
from pred_engine.optimizacion.optimizadores.modelos_clasicos.estrategia import (
    ClassicalSelectionStrategy,
    PresupuestoClasico,
)
from pred_engine.optimizacion.optimizadores.modelos_deep_learning.estrategia import (
    DLSelectionStrategy,
)
from pred_engine.optimizacion.optimizadores.modelos_fundacionales.estrategia import (
    FoundationSelectionStrategy,
)
from pred_engine.optimizacion.optimizadores.modelos_machine_learning.estrategia import (
    MLSelectionStrategy,
    PresupuestoHPO,
)
from pred_engine.optimizacion.router import (
    PREDICTOR_FAMILIES,
    TOPOLOGICAL_PROFILES,
    InitialTopologyPolicy,
    PredictorFamily,
    RoutingDecision,
    SelectionRouter,
    StrategyRegistry,
)
from pred_engine.pipeline import EvaluationSettings, Pipeline, RetrospectiveValidator


class ConfiguredTopologyPolicy:
    """Explicit family subset of the existing matrix, not silent error fallback."""

    def __init__(self, families: tuple[PredictorFamily, ...]) -> None:
        if not families or len(set(families)) != len(families):
            raise ValueError("Select at least one family, without duplicates")
        if not set(families) <= set(PREDICTOR_FAMILIES):
            raise ValueError("Unknown predictor family")
        self.families = families
        self._base = InitialTopologyPolicy()
        self.version = self._base.version + ":configured:" + ",".join(families)

    def decide(self, sku_class: SkuClass) -> tuple[RoutingDecision, ...]:
        return tuple(
            decision
            for decision in self._base.decide(sku_class)
            if decision.family in self.families
        )


def model_factories() -> Mapping[PredictorFamily, FabricaPronosticador]:
    """Real factories, including optional Chronos (weights loaded only on predict)."""
    return {
        "classical": fabrica_sarima,
        "ml": fabrica_ml,
        "dl": fabrica_dl,
        "foundation": fabrica_fundacional,
    }


def build_pipeline(
    evaluation: EvaluationSettings,
    *,
    families: tuple[PredictorFamily, ...] = ("classical", "ml", "dl"),
    n_trials: int | None = None,
    seed: int = 0,
    validator: RetrospectiveValidator | None = None,
    workers: int = 1,
    hpo_root: str | Path | None = None,
    session: str | None = None,
) -> Pipeline:
    """Compose core families; foundation is opt-in and requires the foundation extra.

    With n_trials=None the original strategy budgets are preserved. Evaluation
    settings also define HPO windows. For custom search spaces, construct a
    SelectionRouter with existing strategies and pass it to Pipeline.

    Con ``hpo_root`` y ``session`` cada estudio de HPO persiste su manifiesto y
    sus trials (2.9): repetir la corrida con la misma sesion retoma los estudios
    interrumpidos y reconstruye los completados sin reentrenar.
    """
    policy = ConfiguredTopologyPolicy(families)
    if n_trials is not None and (type(n_trials) is not int or n_trials < 1):
        raise ValueError("n_trials must be a positive integer")
    if (hpo_root is None) != (session is None):
        raise ValueError("hpo_root and session must be given together")
    shared = {
        "min_train": evaluation.min_train,
        "horizonte": evaluation.horizon,
        "paso": evaluation.step,
        "metrica_objetivo": evaluation.metric,
        "seed": seed,
        "raiz_corrida": hpo_root,
        "sesion": session,
    }
    registry = StrategyRegistry()
    for family in families:
        if family == "classical":
            registry.register(
                family,
                ClassicalSelectionStrategy(
                    **shared,
                    presupuestos=None
                    if n_trials is None
                    else {
                        profile: PresupuestoClasico(n_trials=n_trials)
                        for profile in TOPOLOGICAL_PROFILES
                    },
                ),
            )
        elif family == "ml":
            registry.register(
                family,
                MLSelectionStrategy(
                    **shared,
                    presupuestos=None
                    if n_trials is None
                    else {
                        profile: PresupuestoHPO(n_trials=n_trials)
                        for profile in TOPOLOGICAL_PROFILES
                    },
                ),
            )
        elif family == "dl":
            registry.register(
                family,
                DLSelectionStrategy(
                    **shared,
                    n_trials=12 if n_trials is None else n_trials,
                    estacionalidad=evaluation.seasonality,
                ),
            )
        else:
            registry.register(family, FoundationSelectionStrategy())
    return Pipeline(
        SelectionRouter(policy, registry),
        evaluation,
        factories=model_factories(),
        validator=validator,
        workers=workers,
    )
