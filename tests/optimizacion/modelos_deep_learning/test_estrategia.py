"""Resultado comun y despacho real del router sin cambiar su politica inicial."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from tests.optimizacion.modelos_deep_learning.test_dl_selection import CHICO, serie
from tests.optimizacion.router.conftest import FakeStrategy

from pred_engine.comun.modelos import ClassifiedObservation
from pred_engine.optimizacion.optimizadores.modelos_deep_learning import (
    DLSelectionStrategy,
)
from pred_engine.optimizacion.optimizadores.modelos_deep_learning import (
    estrategia as modulo,
)
from pred_engine.optimizacion.router import (
    PREDICTOR_FAMILIES,
    InitialTopologyPolicy,
    SelectionContractError,
    SelectionRequest,
    SelectionResult,
    SelectionRouter,
    SelectionStrategy,
    StrategyRegistry,
)


def solicitud(sku_class="smooth", n=40, sku_id="S1"):
    return SelectionRequest(
        sku_id=sku_id,
        sku_class=sku_class,
        series=tuple(
            ClassifiedObservation(
                sku_id=sku_id,
                sku_class=sku_class,
                timestamp=datetime(2024, 1, 1) + timedelta(days=i),
                demand_qty=float(valor),
                lead_time_days=3,
            )
            for i, valor in enumerate(serie(n))
        ),
    )


def estrategia(**kwargs):
    return DLSelectionStrategy(
        **(dict(espacio=CHICO, n_trials=3, min_train=10, horizonte=2, paso=5) | kwargs)
    )


@pytest.mark.parametrize(
    "clase,perfil",
    [
        ("smooth", "dense_stable"),
        ("erratic", "dense_variable"),
    ],
)
def test_router_invoca_dl_real_con_resultado_comun(clase, perfil):
    dl = estrategia(metrica_objetivo="mae")
    assert isinstance(dl, SelectionStrategy)
    registro = StrategyRegistry()
    for familia in PREDICTOR_FAMILIES:
        registro.register(familia, dl if familia == "dl" else FakeStrategy(familia))
    router = SelectionRouter(InitialTopologyPolicy(), registro)
    resultados = router.route(solicitud(clase))
    assert {r.family for r in resultados} == {"classical", "ml", "dl", "foundation"}
    resultado = next(r for r in resultados if r.family == "dl")
    assert isinstance(resultado, SelectionResult)
    assert resultado.profile == perfil
    assert resultado.sku_id == "S1"
    assert resultado.sku_class == clase
    assert resultado.produced_by == "DLSelectionStrategy"
    assert resultado.policy_version == InitialTopologyPolicy.version
    payload = resultado.payload
    assert payload["metrica_objetivo"] == "mae"
    assert np.isfinite(payload["valor"])
    assert payload["n_ventanas"] == 6
    assert payload["n_trials"] == 3
    assert payload["n_completados"] + payload["n_podados"] + payload["n_fallidos"] == 3
    assert payload["hiperparametros"] == payload["estudio"].mejor.configuracion


@pytest.mark.parametrize("clase", ["intermittent", "lumpy"])
def test_router_no_requiere_ni_invoca_dl_para_clases_excluidas(clase):
    registro = StrategyRegistry()
    for familia in PREDICTOR_FAMILIES:
        if familia != "dl":
            registro.register(familia, FakeStrategy(familia))
    resultados = SelectionRouter(InitialTopologyPolicy(), registro).route(
        solicitud(clase)
    )
    assert "dl" not in {r.family for r in resultados}


def test_serie_se_ordena_y_errores_se_traducen_sin_ocultar_la_causa(monkeypatch):
    original = solicitud()
    invertida = original.model_copy(update={"series": original.series[::-1]})

    def selector(y, **kw):
        np.testing.assert_array_equal(y, serie())
        assert kw["sku_id"] == original.sku_id
        assert kw["n_trials"] == 3
        raise ValueError("causa de origen")

    monkeypatch.setattr(modulo, "seleccionar_configuracion_dl", selector)
    with pytest.raises(SelectionContractError, match="causa de origen") as info:
        estrategia().select(invertida, "dense_stable")
    assert isinstance(info.value.__cause__, ValueError)


def test_sin_ganador_no_devuelve_resultado_vacio(monkeypatch):
    monkeypatch.setattr(
        modulo,
        "seleccionar_configuracion_dl",
        lambda *a, **k: SimpleNamespace(mejor=None, n_fallidos=2, n_podados=1),
    )
    with pytest.raises(SelectionContractError, match="2 fallidos, 1 podados"):
        estrategia().select(solicitud(), "dense_stable")


def test_series_sin_historia_fallan_cerrado():
    vacia = SelectionRequest(sku_id="S1", sku_class="smooth")
    with pytest.raises(SelectionContractError, match="no trae serie"):
        estrategia().select(vacia, "dense_stable")
    with pytest.raises(SelectionContractError, match="insuficiente"):
        estrategia().select(solicitud(n=20), "dense_stable")


def test_persistencia_exige_sesion_y_usa_identidad_estable_segura(tmp_path):
    with pytest.raises(ValueError, match="sesion"):
        estrategia(raiz_corrida=tmp_path)
    dl = estrategia(raiz_corrida=tmp_path, sesion="../sesion")
    request = solicitud(sku_id="../../producto")
    a = dl.select(request, "dense_stable")
    b = dl.select(request, "dense_stable")
    assert a.payload["hiperparametros"] == b.payload["hiperparametros"]
    assert a.payload["valor"] == b.payload["valor"]
    directorios = list(tmp_path.iterdir())
    assert len(directorios) == 1
    assert directorios[0].name.startswith("dl-")
    assert (directorios[0] / "manifiesto.json").is_file()
