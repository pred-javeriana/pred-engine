"""Politica 2.2-A2: matriz declarativa, determinismo, rechazo de clases desconocidas."""

from __future__ import annotations

import pytest

from pred_engine.comun.modelos import SKU_CLASSES
from pred_engine.optimizacion.router import (
    FAMILIES_BY_SKU_CLASS,
    INITIAL_POLICY_VERSION,
    InitialTopologyPolicy,
    PROFILE_BY_SKU_CLASS,
    RoutingPolicy,
    UnknownSkuClassError,
)

ESPERADO = {
    "smooth": (("classical", "ml", "dl", "foundation"), "dense_stable"),
    "erratic": (("classical", "ml", "dl", "foundation"), "dense_variable"),
    "intermittent": (("classical", "ml", "foundation"), "sparse_stable"),
    "lumpy": (("classical", "foundation"), "sparse_variable"),
}


@pytest.mark.parametrize("sku_class", list(SKU_CLASSES))
def test_matriz_por_clase(sku_class: str) -> None:
    politica = InitialTopologyPolicy()
    familias, perfil = ESPERADO[sku_class]
    decisiones = politica.decide(sku_class)  # type: ignore[arg-type]
    assert tuple(d.family for d in decisiones) == familias
    assert {d.profile for d in decisiones} == {perfil}
    assert PROFILE_BY_SKU_CLASS[sku_class] == perfil  # type: ignore[index]
    assert FAMILIES_BY_SKU_CLASS[sku_class] == familias  # type: ignore[index]


def test_politica_satisface_el_contrato() -> None:
    assert isinstance(InitialTopologyPolicy(), RoutingPolicy)
    assert InitialTopologyPolicy.version == INITIAL_POLICY_VERSION


def test_mismas_entradas_mismas_decisiones() -> None:
    politica = InitialTopologyPolicy()
    for clase in SKU_CLASSES:
        assert politica.decide(clase) == politica.decide(clase)


def test_rechaza_clase_desconocida_sin_fallback() -> None:
    politica = InitialTopologyPolicy()
    with pytest.raises(UnknownSkuClassError, match="no soportada"):
        politica.decide("irregular")  # type: ignore[arg-type]
    with pytest.raises(UnknownSkuClassError):
        politica.decide("Smooth")  # type: ignore[arg-type]


def test_intermittent_excluye_dl_lumpy_excluye_ml_y_dl() -> None:
    politica = InitialTopologyPolicy()
    assert "dl" not in {d.family for d in politica.decide("intermittent")}
    lumpy = {d.family for d in politica.decide("lumpy")}
    assert lumpy == {"classical", "foundation"}
