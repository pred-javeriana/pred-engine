"""Enrutador de seleccion sin estado: politica + registro, sin HPO ni I/O."""

from __future__ import annotations

from pred_engine.comun.dataclasses.validacion_temporal import ResultadoWalkForward
from pred_engine.comun.logger import get_logger
from pred_engine.optimizacion.router.contratos import (
    PredictorFamily,
    RoutingDecision,
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

    @property
    def policy(self) -> RoutingPolicy:
        """La politica con la que se enruta; solo lectura."""
        return self._policy

    def route(self, request: SelectionRequest) -> tuple[SelectionResult, ...]:
        return tuple(self.execute(request, decision) for decision in self.plan(request))

    def plan(self, request: SelectionRequest) -> tuple[RoutingDecision, ...]:
        """Decisiones de la politica para la solicitud, sin ejecutar estrategias.

        Separado de `execute` para que cada decision (SKU x familia) pueda
        correr como unidad independiente, incluso en otro proceso.
        """
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
        return tuple(decisiones)

    def execute(
        self, request: SelectionRequest, decision: RoutingDecision
    ) -> SelectionResult:
        """Delegacion de una decision a la estrategia registrada de su familia."""
        return self.execute_with_walk_forward(request, decision)[0]

    def execute_with_walk_forward(
        self, request: SelectionRequest, decision: RoutingDecision
    ) -> tuple[SelectionResult, ResultadoWalkForward | None]:
        """``execute`` y el walk-forward del ganador si la estrategia lo entrega.

        Solo viaja aparte de la seleccion: ``SelectionResult`` se compara y se
        serializa, y este resultado no debe hacer ninguna de las dos cosas.
        """
        if not isinstance(request, SelectionRequest):
            raise SelectionContractError("request debe ser SelectionRequest")
        estrategia = self._registry.resolve(decision.family)
        _logger.info(
            "Enrutando sku_id=%s clase=%s familia=%s perfil=%s politica=%s",
            request.sku_id,
            request.sku_class,
            decision.family,
            decision.profile,
            self._policy.version,
        )
        seleccionar = getattr(estrategia, "select_with_walk_forward", None)
        if seleccionar is None:
            crudo, walk_forward = estrategia.select(request, decision.profile), None
        else:
            crudo, walk_forward = seleccionar(request, decision.profile)
        anotada = self._anotar(request, decision.family, decision.profile, crudo)
        return anotada, walk_forward

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
