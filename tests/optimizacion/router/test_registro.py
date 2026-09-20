"""Registro 2.1-A2: resolver por familia, rechazar ausentes, sustituir con replace."""

from __future__ import annotations

import pytest
from tests.optimizacion.router.conftest import FakeStrategy, make_request

from pred_engine.optimizacion.router import (
    DuplicateStrategyError,
    SelectionContractError,
    SelectionStrategy,
    StrategyRegistry,
    UnregisteredFamilyError,
)


def test_registrar_y_resolver_sin_conocer_la_clase_concreta() -> None:
    registro = StrategyRegistry()
    fake = FakeStrategy("ml", name="HpoMock")
    registro.register("ml", fake)
    resuelta: SelectionStrategy = registro.resolve("ml")
    resultado = resuelta.select(make_request("erratic"), "dense_variable")
    assert resultado.produced_by == "HpoMock"
    assert resultado.family == "ml"
    assert "ml" in registro.families


def test_rechaza_familia_no_registrada() -> None:
    registro = StrategyRegistry()
    registro.register("classical", FakeStrategy("classical"))
    with pytest.raises(UnregisteredFamilyError, match="foundation"):
        registro.resolve("foundation")


def test_rechaza_familia_literal_invalida() -> None:
    registro = StrategyRegistry()
    with pytest.raises(SelectionContractError, match="invalida"):
        registro.register("clasicos", FakeStrategy("classical"))  # type: ignore[arg-type]


def test_rechaza_desalineacion_clave_vs_strategy_family() -> None:
    registro = StrategyRegistry()
    with pytest.raises(SelectionContractError, match="declara family"):
        registro.register("ml", FakeStrategy("dl"))


def test_rechaza_overwrite_silencioso() -> None:
    registro = StrategyRegistry()
    registro.register("dl", FakeStrategy("dl", name="A"))
    with pytest.raises(DuplicateStrategyError):
        registro.register("dl", FakeStrategy("dl", name="B"))
    assert (
        registro.resolve("dl").select(make_request(), "dense_stable").produced_by == "A"
    )


def test_replace_true_sustituye_la_estrategia() -> None:
    registro = StrategyRegistry()
    registro.register("foundation", FakeStrategy("foundation", name="TimesFM"))
    registro.register(
        "foundation", FakeStrategy("foundation", name="Chronos"), replace=True
    )
    out = registro.resolve("foundation").select(
        make_request("lumpy"), "sparse_variable"
    )
    assert out.produced_by == "Chronos"


def test_registro_no_consulta_politica() -> None:
    # Invariante: el modulo de registro no debe importar la politica.
    import pred_engine.optimizacion.router.registro as mod

    assert "politica" not in mod.__name__
    source = __import__("inspect").getsource(mod)
    assert "InitialTopologyPolicy" not in source
    assert "RoutingPolicy" not in source
