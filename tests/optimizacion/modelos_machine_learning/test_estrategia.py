"""MLSelectionStrategy: contrato del router, payload, presupuesto y fail-closed."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from tests.optimizacion.router.conftest import FakeStrategy

from pred_engine.comun.modelos import ClassifiedObservation
from pred_engine.optimizacion.optimizadores.modelos_machine_learning import (
    PRESUPUESTO_POR_PERFIL,
    EspacioML,
    MLSelectionStrategy,
    PresupuestoHPO,
)
from pred_engine.optimizacion.optimizadores.modelos_machine_learning import (
    estrategia as modulo_estrategia,
)
from pred_engine.optimizacion.router import (
    PREDICTOR_FAMILIES,
    InitialTopologyPolicy,
    SelectionContractError,
    SelectionRequest,
    SelectionRouter,
    SelectionStrategy,
    StrategyRegistry,
)

_CHICO = EspacioML(
    lags_min=3,
    lags_max=5,
    n_estimators_min=5,
    n_estimators_max=15,
    max_depth_min=2,
    max_depth_max=3,
)


def _solicitud(
    sku_id: str = "S1",
    sku_class: str = "smooth",
    n: int = 60,
    barajar: bool = False,
) -> SelectionRequest:
    rng = np.random.default_rng(0)
    y = np.maximum(0.0, 10 + 5 * np.sin(np.arange(n) * 2 * np.pi / 7)) + rng.normal(
        0, 1, n
    ).clip(-3, 3)
    y = np.maximum(y, 0.0)
    t0 = datetime(2024, 1, 1)
    obs = [
        ClassifiedObservation(
            sku_id=sku_id,
            timestamp=t0 + timedelta(days=i),
            demand_qty=float(v),
            lead_time_days=3,
            sku_class=sku_class,  # type: ignore[arg-type]
        )
        for i, v in enumerate(y)
    ]
    if barajar:
        obs = obs[::-1]
    return SelectionRequest(
        sku_id=sku_id,
        sku_class=sku_class,  # type: ignore[arg-type]
        series=tuple(obs),
    )


def _estrategia(**kw) -> MLSelectionStrategy:
    kw.setdefault("espacio", _CHICO)
    kw.setdefault("presupuestos", {"dense_stable": PresupuestoHPO(n_trials=5)})
    return MLSelectionStrategy(**kw)


def test_satisface_el_contrato_selection_strategy():
    estrategia = _estrategia()
    assert isinstance(estrategia, SelectionStrategy)
    assert estrategia.family == "ml"


def test_select_devuelve_resultado_con_payload_completo():
    resultado = _estrategia().select(_solicitud(), "dense_stable")
    assert resultado.family == "ml"
    assert resultado.profile == "dense_stable"
    assert resultado.sku_id == "S1"
    assert resultado.produced_by == "MLSelectionStrategy"
    payload = resultado.payload
    assert {"lags", "n_estimators", "max_depth"} <= set(payload["hiperparametros"])
    assert payload["metrica_objetivo"] == "mase"
    assert payload["valor"] > 0
    assert payload["n_trials"] == 5
    assert (
        payload["n_completados"] + payload["n_podados"] + payload["n_fallidos"]
        == payload["n_trials"]
    )


def test_el_presupuesto_depende_del_perfil_topologico():
    estrategia = _estrategia(
        presupuestos={
            "dense_stable": PresupuestoHPO(n_trials=4),
            "dense_variable": PresupuestoHPO(n_trials=7),
        }
    )
    estable = estrategia.select(_solicitud(), "dense_stable")
    variable = estrategia.select(_solicitud(sku_class="erratic"), "dense_variable")
    assert estable.payload["n_trials"] == 4
    assert variable.payload["n_trials"] == 7


def test_presupuestos_por_defecto_cubren_los_cuatro_perfiles():
    assert set(PRESUPUESTO_POR_PERFIL) == {
        "dense_stable",
        "dense_variable",
        "sparse_stable",
        "sparse_variable",
    }
    assert all(p.n_trials > 0 for p in PRESUPUESTO_POR_PERFIL.values())


def test_perfil_sin_presupuesto_falla_cerrado():
    with pytest.raises(SelectionContractError, match="sparse_stable"):
        _estrategia().select(_solicitud(), "sparse_stable")


def test_la_serie_se_ordena_por_timestamp_antes_de_optimizar(monkeypatch):
    capturado = {}

    def _falso(serie, **kw):
        capturado["serie"] = serie
        return SimpleNamespace(
            seleccionada=None,
            estudio=SimpleNamespace(n_fallidos=0, n_podados=0),
        )

    monkeypatch.setattr(modulo_estrategia, "seleccionar_configuracion_ml", _falso)
    ordenada = _solicitud()
    with pytest.raises(SelectionContractError):
        _estrategia().select(_solicitud(barajar=True), "dense_stable")
    esperado = np.array([o.demand_qty for o in ordenada.series])
    np.testing.assert_array_equal(capturado["serie"], esperado)


def test_sin_ningun_trial_completado_falla_cerrado(monkeypatch):
    monkeypatch.setattr(
        modulo_estrategia,
        "seleccionar_configuracion_ml",
        lambda serie, **kw: SimpleNamespace(
            seleccionada=None,
            estudio=SimpleNamespace(n_fallidos=5, n_podados=0),
        ),
    )
    with pytest.raises(SelectionContractError, match="ningun trial"):
        _estrategia().select(_solicitud(), "dense_stable")


def test_solicitud_sin_serie_falla_cerrado():
    vacia = SelectionRequest(sku_id="S1", sku_class="smooth")
    with pytest.raises(SelectionContractError, match="no trae serie"):
        _estrategia().select(vacia, "dense_stable")


def test_serie_insuficiente_se_traduce_a_error_del_router_con_causa():
    with pytest.raises(SelectionContractError, match="insuficiente") as info:
        _estrategia().select(_solicitud(n=10), "dense_stable")
    assert isinstance(info.value.__cause__, ValueError)


def test_raiz_corrida_exige_sesion():
    with pytest.raises(ValueError, match="sesion"):
        MLSelectionStrategy(raiz_corrida="/tmp/x")


def test_run_id_estable_persiste_y_una_segunda_llamada_no_reejecuta(tmp_path):
    estrategia = _estrategia(raiz_corrida=tmp_path, sesion="ses1")
    primera = estrategia.select(_solicitud(), "dense_stable")
    assert (tmp_path / "ml-S1-ses1" / "manifiesto.json").is_file()
    segunda = estrategia.select(_solicitud(), "dense_stable")
    assert segunda.payload["hiperparametros"] == primera.payload["hiperparametros"]
    assert segunda.payload["valor"] == primera.payload["valor"]


def _router_con_ml(estrategia: MLSelectionStrategy) -> SelectionRouter:
    registro = StrategyRegistry()
    for familia in PREDICTOR_FAMILIES:
        if familia == "ml":
            registro.register("ml", estrategia)
        else:
            registro.register(familia, FakeStrategy(familia))
    return SelectionRouter(InitialTopologyPolicy(), registro)


def test_el_router_despacha_smooth_a_la_familia_ml():
    router = _router_con_ml(_estrategia())
    resultados = router.route(_solicitud(sku_class="smooth"))
    por_familia = {r.family: r for r in resultados}
    assert set(por_familia) == {"classical", "ml", "dl", "foundation"}
    ml = por_familia["ml"]
    assert ml.produced_by == "MLSelectionStrategy"
    assert ml.policy_version == InitialTopologyPolicy.version
    assert "hiperparametros" in ml.payload


def test_la_politica_2_2_no_enruta_lumpy_a_ml():
    # Cobertura por clase: ml aplica a smooth/erratic/intermittent; lumpy queda
    # fuera por politica (no por la estrategia).
    router = _router_con_ml(_estrategia())
    resultados = router.route(_solicitud(sku_class="lumpy"))
    assert "ml" not in {r.family for r in resultados}


@pytest.mark.parametrize(
    ("sku_class", "perfil"),
    [
        ("smooth", "dense_stable"),
        ("erratic", "dense_variable"),
        ("intermittent", "sparse_stable"),
    ],
)
def test_ml_funciona_en_las_tres_clases_que_la_politica_le_asigna(sku_class, perfil):
    estrategia = _estrategia(presupuestos={perfil: PresupuestoHPO(n_trials=4)})
    resultado = estrategia.select(_solicitud(sku_class=sku_class), perfil)
    assert resultado.family == "ml"
    assert resultado.sku_class == sku_class
    assert resultado.profile == perfil
