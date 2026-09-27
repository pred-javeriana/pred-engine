"""HPO + Walk-Forward + MLP reales y fronteras compartidas de poda/reanudacion."""

from dataclasses import replace

import numpy as np
import pytest

from pred_engine.comun.modelos.modelos_deep_learning import MLPForecaster
from pred_engine.optimizacion.control_reanudacion import (
    AlmacenManifiestosFs,
    EstadoCorrida,
    IncompatibilidadCorridaError,
)
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda
from pred_engine.optimizacion.optimizadores.modelos_deep_learning import (
    REGLAS_DL,
    EspacioDL,
    seleccionar_configuracion_dl,
)
from pred_engine.optimizacion.optimizadores.modelos_deep_learning import (
    dl_selection as modulo,
)

CHICO = EspacioDL(lags=(3, 4), unidades=(4, 8), epochs=(4, 8))
KW = dict(espacio=CHICO, n_trials=3, min_train=10, horizonte=2, paso=5, seed=0)


def serie(n=40):
    return 10 + np.sin(np.arange(n) * 2 * np.pi / 7)


def test_seleccion_real_tpe_asha_y_prefijos_causales(monkeypatch):
    ajustes = []
    original = MLPForecaster.fit

    def fit(self, y):
        ajustes.append(y.copy())
        return original(self, y)

    monkeypatch.setattr(MLPForecaster, "fit", fit)
    y = serie()
    antes = y.copy()
    resultado = seleccionar_configuracion_dl(y, sku_id="S1", **KW)
    assert resultado.mejor is not None
    assert resultado.mejor.familia == "dl"
    assert resultado.mejor.sku_id == "S1"
    assert np.isfinite(resultado.mejor.valor)
    assert resultado.mejor.n_ventanas == 6
    assert resultado.mejor.estado == "completado"
    assert len(resultado.trials) == 3
    assert len(resultado.ventanas) == 6
    for ventana in resultado.ventanas:
        assert ventana.inicio_train == 0
        assert ventana.fin_train == ventana.inicio_val
        assert ventana.fin_val - ventana.inicio_val == 2
    assert len(ajustes) == sum(t.n_ventanas for t in resultado.trials)
    for prefijo in ajustes:
        assert len(prefijo) in {v.fin_train for v in resultado.ventanas}
        np.testing.assert_array_equal(prefijo, y[: len(prefijo)])
    np.testing.assert_array_equal(y, antes)
    assert all(t.n_ventanas >= 4 for t in resultado.trials if t.estado == "podado")


def test_semilla_reproduce_configuracion_y_valor_y_acepta_metricas_compartidas():
    a = seleccionar_configuracion_dl(serie(), **KW, metrica_objetivo="mae")
    b = seleccionar_configuracion_dl(serie(), **KW, metrica_objetivo="mae")
    assert a.mejor.configuracion == b.mejor.configuracion
    assert a.mejor.valor == b.mejor.valor
    assert a.metrica_objetivo == "mae"


def test_podados_y_fallos_de_entrenamiento_permanecen_distintos(monkeypatch):
    indices = {}

    class Candidato:
        def __init__(self, indice):
            self.indice = indice

        def fit(self, y):
            if self.indice == 2:
                raise RuntimeError("fallo de entrenamiento")
            return self

        def predict(self, horizon):
            return np.full(horizon, 10.0 if self.indice == 0 else 100.0)

    def fabrica(configuracion, *, seed):
        clave = tuple(configuracion.items())
        indice = indices.setdefault(clave, len(indices))
        return Candidato(indice)

    monkeypatch.setattr(modulo, "fabrica_dl", fabrica)
    estudio = seleccionar_configuracion_dl(serie(), **KW)
    assert [t.estado for t in estudio.trials] == ["completado", "podado", "fallido"]
    assert (estudio.n_completados, estudio.n_podados, estudio.n_fallidos) == (1, 1, 1)
    assert estudio.trials[1].n_ventanas == 4
    assert estudio.trials[1].motivo
    assert estudio.trials[2].motivo
    assert estudio.mejor == estudio.trials[0]


@pytest.mark.parametrize(
    "y,kw,mensaje",
    [
        (np.ones((40, 2)), {}, "1D"),
        (np.full(40, np.nan), {}, "no finitos"),
        (serie(5), {}, "insuficiente"),
        (serie(20), {}, "ventanas"),
        (serie(), {"min_train": 8}, "min_train insuficiente"),
        (serie(), {"n_trials": 0}, "n_trials"),
        (serie(), {"paso": 0}, "paso"),
        (serie(), {"horizonte": 0}, "horizonte"),
        (serie(), {"estacionalidad": 0}, "estacionalidad"),
        (serie(), {"estacionalidad": 10}, "MASE"),
        (serie(), {"reglas": replace(REGLAS_DL, min_ventanas=3)}, "min_ventanas"),
        (
            serie(),
            {"reglas": replace(REGLAS_DL, factor_reduccion=1)},
            "factor_reduccion",
        ),
        (serie(), {"reglas": ReglasPoda()}, "poda_semantica=False"),
    ],
)
def test_entradas_invalidas_fallan_antes_de_entrenar(monkeypatch, y, kw, mensaje):
    def prohibida(*a, **k):
        pytest.fail("no debe iniciar HPO")

    monkeypatch.setattr(modulo, "seleccionar_con_hpo", prohibida)
    with pytest.raises(ValueError, match=mensaje):
        seleccionar_configuracion_dl(y, **(KW | kw))


def test_reutiliza_controlador_y_rechaza_cambio_de_serie_o_cota(tmp_path, monkeypatch):
    kw = KW | dict(sku_id="S1", raiz_corrida=tmp_path, run_id="dl-test")
    a = seleccionar_configuracion_dl(serie(), **kw)
    assert (
        AlmacenManifiestosFs(tmp_path).cargar("dl-test").estado
        is EstadoCorrida.COMPLETADA
    )

    def prohibida(*a, **k):
        pytest.fail("una corrida completada no vuelve a entrenar")

    monkeypatch.setattr(modulo, "fabrica_dl", prohibida)
    b = seleccionar_configuracion_dl(serie(), **kw)
    assert b.mejor.configuracion == a.mejor.configuracion
    assert b.mejor.valor == a.mejor.valor
    assert [t.estado for t in a.trials] == [t.estado for t in b.trials]
    with pytest.raises(IncompatibilidadCorridaError):
        seleccionar_configuracion_dl(serie() + 1, **kw)
    with pytest.raises(IncompatibilidadCorridaError):
        seleccionar_configuracion_dl(
            serie(), **(kw | {"espacio": replace(CHICO, costo_max=90_000)})
        )


def test_interrupcion_se_reanuda_con_controlador_existente(tmp_path, monkeypatch):
    original = modulo.fabrica_dl
    llamadas = 0

    def interrumpir(configuracion, *, seed):
        nonlocal llamadas
        llamadas += 1
        if llamadas == 7:
            raise KeyboardInterrupt
        return original(configuracion, seed=seed)

    kw = KW | dict(sku_id="S1", raiz_corrida=tmp_path, run_id="dl-interrumpida")
    monkeypatch.setattr(modulo, "fabrica_dl", interrumpir)
    with pytest.raises(KeyboardInterrupt):
        seleccionar_configuracion_dl(serie(), **kw)
    almacen = AlmacenManifiestosFs(tmp_path)
    assert almacen.cargar("dl-interrumpida").estado is EstadoCorrida.INTERRUMPIDA
    monkeypatch.setattr(modulo, "fabrica_dl", original)
    estudio = seleccionar_configuracion_dl(serie(), **kw)
    assert estudio.mejor is not None
    assert almacen.cargar("dl-interrumpida").estado is EstadoCorrida.COMPLETADA
