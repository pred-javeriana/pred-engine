"""Motor de enrutamiento del Modulo 2: Router + Strategy sin estado."""

from pred_engine.optimizacion.router.contratos import (
    PREDICTOR_FAMILIES,
    TOPOLOGICAL_PROFILES,
    PredictorFamily,
    RoutingDecision,
    RoutingPolicy,
    SelectionRequest,
    SelectionResult,
    SelectionStrategy,
    TopologicalProfile,
)
from pred_engine.optimizacion.router.errores import (
    DuplicateStrategyError,
    RouterConfigurationError,
    SelectionContractError,
    SelectionError,
    UnknownSkuClassError,
    UnregisteredFamilyError,
)
from pred_engine.optimizacion.router.registro import StrategyRegistry

__all__ = [
    "DuplicateStrategyError",
    "PREDICTOR_FAMILIES",
    "PredictorFamily",
    "RoutingDecision",
    "RoutingPolicy",
    "RouterConfigurationError",
    "SelectionContractError",
    "SelectionError",
    "SelectionRequest",
    "SelectionResult",
    "SelectionStrategy",
    "StrategyRegistry",
    "TOPOLOGICAL_PROFILES",
    "TopologicalProfile",
    "UnknownSkuClassError",
    "UnregisteredFamilyError",
]
