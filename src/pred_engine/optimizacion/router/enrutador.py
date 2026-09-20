"""Enrutador de seleccion sin estado: politica + registro, sin HPO ni I/O."""

from __future__ import annotations

from pred_engine.comun.logger import get_logger
from pred_engine.optimizacion.router.contratos import (
    PredictorFamily,
    RoutingPolicy,
    SelectionRequest,
    SelectionResult,
    TopologicalProfile,
)
from pred_engine.optimizacion.router.errores import (
    RouterConfigurationError,
    SelectionContractError,
)
from pred_engine.optimizacion.router.registro import StrategyRegistry

_logger = get_logger(__name__)


class SelectionRouter:
    """Coordina enrutamiento y delegacion. No retiene estado entre invocaciones."""

    def __init__(self, policy: RoutingPolicy, registry: StrategyRegistry) -> None:
        if not isinstance(policy, RoutingPolicy):
            raise RouterConfigurationError(
                "policy debe satisfacer el contrato RoutingPolicy"
            )
        if not isinstance(registry, StrategyRegistry):
            raise RouterConfigurationError("registry debe ser un StrategyRegistry")
        if not str(policy.version).strip():
            raise RouterConfigurationError("la politica debe declarar version")
        self._policy = policy
        self._registry = registry

    def route(self, request: SelectionRequest) -> tuple[SelectionResult, ...]:
        if not isinstance(request, SelectionRequest):
            raise SelectionContractError("request debe ser SelectionRequest")

        decisiones = self._policy.decide(request.sku_class)
        if not decisiones:
            _logger.error(
                "Politica sin decisiones sku_id=%s clase=%s version=%s",
                request.sku_id,
                request.sku_class,
                self._policy.version,
            )
            raise RouterConfigurationError(
                f"la politica {self._policy.version!r} no produjo decisiones"
            )

        resultados: list[SelectionResult] = []
        for decision in decisiones:
            estrategia = self._registry.resolve(decision.family)
            _logger.info(
                "Enrutando sku_id=%s clase=%s familia=%s perfil=%s politica=%s",
                request.sku_id,
                request.sku_class,
                decision.family,
                decision.profile,
                self._policy.version,
            )
            crudo = estrategia.select(request, decision.profile)
            resultados.append(
                self._anotar(request, decision.family, decision.profile, crudo)
            )
        return tuple(resultados)

    def _anotar(
        self,
        request: SelectionRequest,
        familia: PredictorFamily,
        perfil: TopologicalProfile,
        result: SelectionResult,
    ) -> SelectionResult:
        if not isinstance(result, SelectionResult):
            raise SelectionContractError("la estrategia no devolvio SelectionResult")
        if result.sku_id != request.sku_id or result.sku_class != request.sku_class:
            raise SelectionContractError(
                "el resultado no coincide con la identidad de la solicitud"
            )
        if result.family != familia or result.profile != perfil:
            raise SelectionContractError(
                "el resultado no coincide con la decision de enrutamiento"
            )
        if result.policy_version and result.policy_version != self._policy.version:
            raise SelectionContractError(
                "el resultado declara una version de politica distinta"
            )
        if result.policy_version:
            return result
        return result.model_copy(update={"policy_version": self._policy.version})
