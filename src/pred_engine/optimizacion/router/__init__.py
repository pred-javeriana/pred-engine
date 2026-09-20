"""Motor de enrutamiento del Modulo 2: Router + Strategy sin estado."""

from pred_engine.optimizacion.router.contratos import (
    PREDICTOR_FAMILIES,
    TOPOLOGICAL_PROFILES,
    PredictorFamily,
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

__all__ = [
    "DuplicateStrategyError",
    "PREDICTOR_FAMILIES",
    "PredictorFamily",
    "RouterConfigurationError",
    "SelectionContractError",
    "SelectionError",
    "SelectionRequest",
    "SelectionResult",
    "SelectionStrategy",
    "TOPOLOGICAL_PROFILES",
    "TopologicalProfile",
    "UnknownSkuClassError",
    "UnregisteredFamilyError",
]
