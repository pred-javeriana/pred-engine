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
from pred_engine.optimizacion.router.politica import (
    FAMILIES_BY_SKU_CLASS,
    INITIAL_POLICY_VERSION,
    InitialTopologyPolicy,
    PROFILE_BY_SKU_CLASS,
)
from pred_engine.optimizacion.router.enrutador import SelectionRouter
from pred_engine.optimizacion.router.registro import StrategyRegistry

__all__ = [
    "DuplicateStrategyError",
    "FAMILIES_BY_SKU_CLASS",
    "INITIAL_POLICY_VERSION",
    "InitialTopologyPolicy",
    "PREDICTOR_FAMILIES",
    "PROFILE_BY_SKU_CLASS",
    "PredictorFamily",
    "RoutingDecision",
    "RoutingPolicy",
    "RouterConfigurationError",
    "SelectionContractError",
    "SelectionError",
    "SelectionRequest",
    "SelectionResult",
    "SelectionRouter",
    "SelectionStrategy",
    "StrategyRegistry",
    "TOPOLOGICAL_PROFILES",
    "TopologicalProfile",
    "UnknownSkuClassError",
    "UnregisteredFamilyError",
]
