"""FoundationSelectionStrategy: contrato del router y configuracion base."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from tests.optimizacion.router.conftest import FakeStrategy

from pred_engine.comun.modelos import ClassifiedObservation
from pred_engine.comun.modelos.modelos_fundacionales import CHRONOS2_ZERO_SHOT
from pred_engine.comun.modelos.modelos_fundacionales import pipeline as modulo_pipeline
from pred_engine.optimizacion.optimizadores.modelos_fundacionales import (
    FoundationSelectionStrategy,
)
from pred_engine.optimizacion.router import (
    PREDICTOR_FAMILIES,
    TOPOLOGICAL_PROFILES,
    InitialTopologyPolicy,
    SelectionRequest,
    SelectionRouter,
    SelectionStrategy,
    StrategyRegistry,
)


def _solicitud(sku_id: str = "S1", sku_class: str = "smooth", n: int = 30):
    t0 = datetime(2024, 1, 1)
    obs = tuple(
        ClassifiedObservation(
            sku_id=sku_id,
            timestamp=t0 + timedelta(days=i),
            demand_qty=float(i % 3),
            lead_time_days=3,
            sku_class=sku_class,  # type: ignore[arg-type]
        )
        for i in range(n)
    )
    return SelectionRequest(
        sku_id=sku_id,
        sku_class=sku_class,  # type: ignore[arg-type]
        series=obs,
    )


def test_satisface_el_contrato_selection_strategy():
    estrategia = FoundationSelectionStrategy()
    assert isinstance(estrategia, SelectionStrategy)
    assert estrategia.family == "foundation"


def test_select_devuelve_la_configuracion_base():
    resultado = FoundationSelectionStrategy().select(_solicitud(), "dense_stable")
    assert resultado.family == "foundation"
    assert resultado.profile == "dense_stable"
    assert resultado.sku_id == "S1"
    assert resultado.produced_by == "FoundationSelectionStrategy"
    assert resultado.payload == {
        "configuracion": CHRONOS2_ZERO_SHOT.descripcion_canonica(),
        "optimizado": False,
    }


def test_la_configuracion_es_identica_en_los_cuatro_perfiles():
    estrategia = FoundationSelectionStrategy()
    payloads = [
        estrategia.select(_solicitud(), perfil).payload
        for perfil in TOPOLOGICAL_PROFILES
    ]
    assert all(p == payloads[0] for p in payloads)


def test_select_no_necesita_serie_ni_carga_el_modelo(monkeypatch):
    def _prohibido(config):
        raise AssertionError("select() no debe cargar el modelo")

    monkeypatch.setattr(modulo_pipeline, "cargar_pipeline", _prohibido)
    vacia = SelectionRequest(sku_id="S1", sku_class="lumpy")
    resultado = FoundationSelectionStrategy().select(vacia, "sparse_variable")
    assert resultado.payload["optimizado"] is False


def _router_con_fundacional() -> SelectionRouter:
    registro = StrategyRegistry()
    for familia in PREDICTOR_FAMILIES:
        if familia == "foundation":
            registro.register("foundation", FoundationSelectionStrategy())
        else:
            registro.register(familia, FakeStrategy(familia))
    return SelectionRouter(InitialTopologyPolicy(), registro)


@pytest.mark.parametrize(
    ("sku_class", "perfil"),
    [
        ("smooth", "dense_stable"),
        ("erratic", "dense_variable"),
        ("intermittent", "sparse_stable"),
        ("lumpy", "sparse_variable"),
    ],
)
def test_el_router_despacha_las_cuatro_clases_a_la_familia_fundacional(
    sku_class, perfil
):
    resultados = _router_con_fundacional().route(_solicitud(sku_class=sku_class))
    fundacional = {r.family: r for r in resultados}["foundation"]
    assert fundacional.produced_by == "FoundationSelectionStrategy"
    assert fundacional.profile == perfil
    assert fundacional.sku_class == sku_class
    assert fundacional.policy_version == InitialTopologyPolicy.version
