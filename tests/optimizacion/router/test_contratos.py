"""Contratos 2.1-A1: familias cerradas, SkuClass del Modulo 1, sin Optuna."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from pred_engine.comun.modelos import SKU_CLASSES, ClassifiedObservation, SkuClass
from pred_engine.optimizacion.router import (
    PREDICTOR_FAMILIES,
    TOPOLOGICAL_PROFILES,
    RoutingDecision,
    RoutingPolicy,
    SelectionRequest,
    SelectionResult,
    SelectionStrategy,
)


def _observacion(*, sku_id: str = "105", sku_class: str = "intermittent"):
    return ClassifiedObservation(
        sku_id=sku_id,
        timestamp=datetime(2024, 10, 1),
        demand_qty=1.0,
        lead_time_days=17,
        sku_class=sku_class,  # type: ignore[arg-type]
    )


def test_familias_soportadas_son_cuatro_y_cerradas() -> None:
    assert PREDICTOR_FAMILIES == ("classical", "ml", "dl", "foundation")


def test_perfiles_soportados_coinciden_con_la_matriz_2_2() -> None:
    assert TOPOLOGICAL_PROFILES == (
        "dense_stable",
        "dense_variable",
        "sparse_stable",
        "sparse_variable",
    )


def test_solicitud_reutiliza_sku_class_de_ingesta() -> None:
    for clase in SKU_CLASSES:
        req = SelectionRequest(sku_id="105", sku_class=clase)
        assert req.sku_class == clase
        assert req.series == ()


def test_solicitud_rechaza_title_case_de_notion() -> None:
    with pytest.raises(ValidationError):
        SelectionRequest(sku_id="105", sku_class="Smooth")  # type: ignore[arg-type]


def test_solicitud_rechaza_sku_en_blanco() -> None:
    with pytest.raises(ValidationError):
        SelectionRequest(sku_id="   ", sku_class="smooth")


def test_solicitud_rechaza_campos_extra() -> None:
    with pytest.raises(ValidationError):
        SelectionRequest(sku_id="105", sku_class="smooth", optuna_study=1)  # type: ignore[call-arg]


def test_solicitud_acepta_serie_del_contrato_1_4() -> None:
    obs = _observacion()
    req = SelectionRequest(sku_id="105", sku_class="intermittent", series=(obs,))
    assert req.series[0].demand_qty == 1.0


def test_solicitud_rechaza_serie_de_otro_sku() -> None:
    obs = _observacion(sku_id="999")
    with pytest.raises(ValidationError, match="sku_id"):
        SelectionRequest(sku_id="105", sku_class="intermittent", series=(obs,))


def test_solicitud_rechaza_serie_de_otra_clase() -> None:
    obs = _observacion(sku_class="lumpy")
    with pytest.raises(ValidationError, match="sku_class"):
        SelectionRequest(sku_id="105", sku_class="intermittent", series=(obs,))


def test_resultado_rechaza_familia_interna_de_hpo() -> None:
    with pytest.raises(ValidationError):
        SelectionResult(
            sku_id="105",
            sku_class="smooth",
            family="clasicos",  # type: ignore[arg-type]
            profile="dense_stable",
            produced_by="x",
        )


def test_resultado_rechaza_perfil_inventado() -> None:
    with pytest.raises(ValidationError):
        SelectionResult(
            sku_id="105",
            sku_class="smooth",
            family="classical",
            profile="dense",  # type: ignore[arg-type]
            produced_by="x",
        )


def test_resultado_es_inmutable() -> None:
    resultado = SelectionResult(
        sku_id="105",
        sku_class="smooth",
        family="classical",
        profile="dense_stable",
        produced_by="Fake",
    )
    with pytest.raises(ValidationError):
        resultado.sku_id = "otro"  # type: ignore[misc]


def test_estrategia_simulada_satisface_el_protocolo() -> None:
    class Dummy:
        family = "ml"

        def select(self, request: SelectionRequest, profile: str) -> SelectionResult:
            return SelectionResult(
                sku_id=request.sku_id,
                sku_class=request.sku_class,
                family="ml",
                profile=profile,  # type: ignore[arg-type]
                produced_by="Dummy",
            )

    dummy = Dummy()
    assert isinstance(dummy, SelectionStrategy)
    req = SelectionRequest(sku_id="105", sku_class="erratic")
    out = dummy.select(req, "dense_variable")
    assert out.family == "ml"
    assert out.profile == "dense_variable"


def test_decision_es_igual_por_valor_y_es_inmutable() -> None:
    a = RoutingDecision(family="classical", profile="sparse_variable")
    b = RoutingDecision(family="classical", profile="sparse_variable")
    c = RoutingDecision(family="foundation", profile="sparse_variable")
    assert a == b
    assert a != c
    with pytest.raises(ValidationError):
        a.family = "ml"  # type: ignore[misc]


def test_decision_rechaza_familia_o_perfil_invalidos() -> None:
    with pytest.raises(ValidationError):
        RoutingDecision(family="CLASSICAL", profile="dense_stable")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        RoutingDecision(family="ml", profile="sparse")  # type: ignore[arg-type]


def test_politica_simulada_satisface_el_protocolo() -> None:
    class PoliticaFija:
        version = "test-1"

        def decide(self, sku_class: SkuClass) -> tuple[RoutingDecision, ...]:
            return (
                RoutingDecision(family="classical", profile="dense_stable"),
                RoutingDecision(family="foundation", profile="dense_stable"),
            )

    politica = PoliticaFija()
    assert isinstance(politica, RoutingPolicy)
    d1 = politica.decide("smooth")
    d2 = politica.decide("lumpy")
    assert d1 == d2
    assert d1[0].family == "classical"
