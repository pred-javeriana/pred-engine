"""Dobles de prueba que satisfacen SelectionStrategy sin HPO ni modelos."""

from __future__ import annotations

from dataclasses import dataclass, field

from pred_engine.optimizacion.router import (
    PREDICTOR_FAMILIES,
    PredictorFamily,
    SelectionRequest,
    SelectionResult,
    StrategyRegistry,
    TopologicalProfile,
)


@dataclass
class FakeStrategy:
    family: PredictorFamily
    name: str = "FakeStrategy"
    calls: list[tuple[SelectionRequest, TopologicalProfile]] = field(
        default_factory=list
    )

    def select(
        self, request: SelectionRequest, profile: TopologicalProfile
    ) -> SelectionResult:
        self.calls.append((request, profile))
        return SelectionResult(
            sku_id=request.sku_id,
            sku_class=request.sku_class,
            family=self.family,
            profile=profile,
            produced_by=self.name,
        )


def make_request(sku_class: str = "smooth", sku_id: str = "105") -> SelectionRequest:
    return SelectionRequest(sku_id=sku_id, sku_class=sku_class)  # type: ignore[arg-type]


def populated_registry() -> tuple[StrategyRegistry, dict[str, FakeStrategy]]:
    fakes = {familia: FakeStrategy(familia) for familia in PREDICTOR_FAMILIES}
    registro = StrategyRegistry()
    for familia, fake in fakes.items():
        registro.register(familia, fake)
    return registro, fakes
