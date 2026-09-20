"""Enrutador 2.1-B1 / 2.2-B1: delegacion, ausencia de estado, fail-closed."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from pred_engine.comun.modelos import SKU_CLASSES
from pred_engine.optimizacion.router import (
    INITIAL_POLICY_VERSION,
    InitialTopologyPolicy,
    RouterConfigurationError,
    RoutingDecision,
    SelectionContractError,
    SelectionRequest,
    SelectionResult,
    SelectionRouter,
    StrategyRegistry,
    UnregisteredFamilyError,
)
from tests.optimizacion.router.conftest import (
    FakeStrategy,
    make_request,
    populated_registry,
)


class PoliticaVacia:
    version = "empty-0"

    def decide(self, sku_class: object) -> tuple[RoutingDecision, ...]:
        return ()


class PoliticaFija:
    version = "fija-1"

    def decide(self, sku_class: object) -> tuple[RoutingDecision, ...]:
        return (
            RoutingDecision(family="classical", profile="dense_stable"),
            RoutingDecision(family="ml", profile="dense_stable"),
        )


def test_delega_a_todas_las_estrategias_de_la_politica() -> None:
    registro, fakes = populated_registry()
    router = SelectionRouter(PoliticaFija(), registro)
    resultados = router.route(make_request("smooth"))
    assert [r.family for r in resultados] == ["classical", "ml"]
    assert fakes["classical"].calls and fakes["ml"].calls
    assert not fakes["dl"].calls
    assert not fakes["foundation"].calls
    assert all(r.policy_version == "fija-1" for r in resultados)


def test_enrutador_sin_estado_entre_invocaciones() -> None:
    registro, _ = populated_registry()
    router = SelectionRouter(InitialTopologyPolicy(), registro)
    assert set(vars(router)) <= {"_policy", "_registry"}
    primero = router.route(make_request("smooth", sku_id="A"))
    segundo = router.route(make_request("lumpy", sku_id="B"))
    assert set(vars(router)) <= {"_policy", "_registry"}
    assert {r.sku_id for r in primero} == {"A"}
    assert {r.sku_id for r in segundo} == {"B"}
    assert {r.family for r in segundo} == {"classical", "foundation"}


def test_rechaza_request_que_no_es_contrato() -> None:
    registro, _ = populated_registry()
    router = SelectionRouter(InitialTopologyPolicy(), registro)
    with pytest.raises(SelectionContractError, match="SelectionRequest"):
        router.route({"sku_id": "105", "sku_class": "smooth"})  # type: ignore[arg-type]


def test_rechaza_dependencias_invalidas() -> None:
    registro, _ = populated_registry()
    with pytest.raises(RouterConfigurationError, match="policy"):
        SelectionRouter(object(), registro)  # type: ignore[arg-type]
    with pytest.raises(RouterConfigurationError, match="registry"):
        SelectionRouter(InitialTopologyPolicy(), object())  # type: ignore[arg-type]


def test_rechaza_politica_sin_decisiones() -> None:
    registro, _ = populated_registry()
    router = SelectionRouter(PoliticaVacia(), registro)
    with pytest.raises(RouterConfigurationError, match="no produjo decisiones"):
        router.route(make_request())


def test_familia_de_politica_sin_estrategia_falla_explicito() -> None:
    registro = StrategyRegistry()
    registro.register("classical", FakeStrategy("classical"))
    router = SelectionRouter(PoliticaFija(), registro)
    with pytest.raises(UnregisteredFamilyError, match="ml"):
        router.route(make_request())


def test_resultado_desalineado_es_rechazado() -> None:
    class Traidora(FakeStrategy):
        def select(self, request: SelectionRequest, profile: str) -> SelectionResult:
            return SelectionResult(
                sku_id=request.sku_id,
                sku_class=request.sku_class,
                family="dl",
                profile=profile,  # type: ignore[arg-type]
                produced_by="Traidora",
            )

    registro = StrategyRegistry()
    registro.register("classical", Traidora("classical"))
    registro.register("ml", FakeStrategy("ml"))
    router = SelectionRouter(PoliticaFija(), registro)
    with pytest.raises(SelectionContractError, match="decision"):
        router.route(make_request())
