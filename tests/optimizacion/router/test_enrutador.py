"""Enrutador 2.1-B1 / 2.2-B1: delegacion, ausencia de estado, fail-closed."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from tests.optimizacion.router.conftest import (
    FakeStrategy,
    make_request,
    populated_registry,
)

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


@pytest.mark.parametrize(
    ("sku_class", "families", "profile"),
    [
        ("smooth", ("classical", "ml", "dl", "foundation"), "dense_stable"),
        ("erratic", ("classical", "ml", "dl", "foundation"), "dense_variable"),
        ("intermittent", ("classical", "ml", "foundation"), "sparse_stable"),
        ("lumpy", ("classical", "foundation"), "sparse_variable"),
    ],
)
def test_cuatro_clases_reducen_el_espacio(
    sku_class: str, families: tuple[str, ...], profile: str
) -> None:
    registro, fakes = populated_registry()
    router = SelectionRouter(InitialTopologyPolicy(), registro)
    resultados = router.route(make_request(sku_class))
    assert tuple(r.family for r in resultados) == families
    assert {r.profile for r in resultados} == {profile}
    assert all(r.policy_version == INITIAL_POLICY_VERSION for r in resultados)
    for familia, fake in fakes.items():
        if familia in families:
            assert fake.calls, familia
            assert fake.calls[0][1] == profile
        else:
            assert fake.calls == []


def test_estrategia_excluida_no_se_invoca_en_lumpy() -> None:
    registro, fakes = populated_registry()
    SelectionRouter(InitialTopologyPolicy(), registro).route(make_request("lumpy"))
    assert fakes["ml"].calls == []
    assert fakes["dl"].calls == []
    assert len(fakes["classical"].calls) == 1
    assert len(fakes["foundation"].calls) == 1


def test_propaga_perfil_sparse_stable_en_intermittent() -> None:
    registro, fakes = populated_registry()
    SelectionRouter(InitialTopologyPolicy(), registro).route(
        make_request("intermittent")
    )
    for familia in ("classical", "ml", "foundation"):
        assert fakes[familia].calls[0][1] == "sparse_stable"
    assert fakes["dl"].calls == []


def test_decisiones_quedan_registradas(monkeypatch: pytest.MonkeyPatch) -> None:
    eventos: list[str] = []

    class _Captura:
        def info(self, mensaje: str, *args: object) -> None:
            eventos.append(mensaje % args if args else mensaje)

        def error(self, mensaje: str, *args: object) -> None:
            eventos.append(mensaje % args if args else mensaje)

    monkeypatch.setattr("pred_engine.optimizacion.router.enrutador._logger", _Captura())
    registro, _ = populated_registry()
    SelectionRouter(InitialTopologyPolicy(), registro).route(make_request("lumpy"))
    trazas = [e for e in eventos if e.startswith("Enrutando sku_id=105")]
    assert len(trazas) == 2
    assert any(
        "familia=classical" in e and "perfil=sparse_variable" in e for e in trazas
    )
    assert any("familia=foundation" in e for e in trazas)
    assert all("politica=2.2.0-initial" in e for e in trazas)


def test_paquete_router_no_importa_hpo_ni_walkforward() -> None:
    raiz = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "pred_engine"
        / "optimizacion"
        / "router"
    )
    prohibidos = (
        "optuna",
        "walkforward",
        "classical_selection",
        "ejecutar_estudio",
        "asha",
    )
    for path in raiz.glob("*.py"):
        arbol = ast.parse(path.read_text(encoding="utf-8"))
        nombres: set[str] = set()
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Import):
                nombres.update(alias.name for alias in nodo.names)
            elif isinstance(nodo, ast.ImportFrom) and nodo.module:
                nombres.add(nodo.module)
        unidos = " ".join(nombres)
        for palabra in prohibidos:
            assert palabra not in unidos, f"{path.name} importa {palabra}"
