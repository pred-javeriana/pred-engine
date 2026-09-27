"""Entrenamiento real, causalidad y reproducibilidad del predictor DL minimo."""

from __future__ import annotations

import numpy as np
import pytest

from pred_engine.comun.modelos.modelos_deep_learning import MLPForecaster, fabrica_dl
from pred_engine.comun.modelos.modelos_deep_learning.mlp import numero_parametros
from pred_engine.comun.walkforward.protocolos import Pronosticador


def _serie(n=80):
    return 10 + 3 * np.sin(np.arange(n) * 2 * np.pi / 7)


def test_contrato_entrena_pronostica_y_no_muta_entrada_ni_estado_en_predict():
    y = _serie()
    antes = y.copy()
    modelo = fabrica_dl({"lags": 7, "capas": 3, "unidades": 8}, seed=17)
    assert isinstance(modelo, Pronosticador)
    assert modelo.fit(y) is modelo
    pred = modelo.predict(10)
    assert pred.shape == (10,)
    assert np.isfinite(pred).all() and (pred >= 0).all()
    np.testing.assert_array_equal(modelo.predict(10), pred)
    np.testing.assert_array_equal(y, antes)
    assert sum(p.size for p in modelo._pesos + modelo._sesgos) == numero_parametros(
        7, 3, 8
    )


def test_mas_entrenamiento_aprende_patron_y_reduce_error_fuera_de_muestra():
    y = _serie(94)
    inicial = MLPForecaster(lags=7, epochs=1, seed=3).fit(y[:80]).predict(14)
    entrenado = MLPForecaster(lags=7, epochs=120, seed=3).fit(y[:80]).predict(14)
    assert np.mean((entrenado - y[80:]) ** 2) < 0.1 * np.mean((inicial - y[80:]) ** 2)


def test_semilla_local_dropout_y_refit_reproducibles_sin_rng_global():
    np.random.seed(91)
    esperado = np.random.random(3)
    np.random.seed(91)
    modelo = MLPForecaster(dropout=0.2, seed=7)
    a = modelo.fit(_serie()).predict(4)
    b = MLPForecaster(dropout=0.2, seed=7).fit(_serie()).predict(4)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(np.random.random(3), esperado)
    modelo.fit(_serie() + 30)
    np.testing.assert_array_equal(modelo.fit(_serie()).predict(4), a)
    diferente = MLPForecaster(dropout=0.2, seed=8).fit(_serie()).predict(4)
    assert not np.array_equal(a, diferente)


def test_normalizacion_solo_usa_el_prefijo_y_la_constante_es_finita():
    prefijo = _serie(30)
    a = MLPForecaster().fit(prefijo)
    assert a._media == pytest.approx(prefijo.mean())
    assert a._escala == pytest.approx(prefijo.std())
    constante = MLPForecaster().fit(np.full(30, 8.0))
    np.testing.assert_allclose(constante.predict(5), 8.0)


def test_gradientes_coinciden_con_diferencias_finitas_incluyendo_l2_y_dropout():
    modelo = MLPForecaster(lags=2, capas=2, unidades=2, l2=0.1, dropout=0.2)
    modelo.fit(_serie(12))
    x = np.array([[0.2, -0.1], [0.8, 0.4]])
    y = np.array([[0.3], [0.5]])
    gp, gs = modelo._gradientes(x, y, np.random.default_rng(42))

    def perdida():
        rng = np.random.default_rng(42)
        a = x
        for p, s in zip(modelo._pesos[:-1], modelo._sesgos[:-1], strict=True):
            a = np.tanh(a @ p + s)
            a *= (rng.random(a.shape) >= modelo.dropout) / (1 - modelo.dropout)
        pred = a @ modelo._pesos[-1] + modelo._sesgos[-1]
        return np.mean((pred - y) ** 2) + modelo.l2 / 2 * sum(
            (p**2).sum() for p in modelo._pesos
        )

    for parametro, gradiente in zip(
        modelo._pesos + modelo._sesgos, gp + gs, strict=True
    ):
        for indice in np.ndindex(parametro.shape):
            original = parametro[indice]
            parametro[indice] = original + 1e-6
            mas = perdida()
            parametro[indice] = original - 1e-6
            menos = perdida()
            parametro[indice] = original
            assert gradiente[indice] == pytest.approx((mas - menos) / 2e-6, abs=1e-7)


@pytest.mark.parametrize("y", [np.ones(8), np.ones((20, 2)), np.full(20, np.nan)])
def test_fit_invalido_inhabilita_un_modelo_ajustado(y):
    modelo = MLPForecaster().fit(_serie())
    with pytest.raises(ValueError):
        modelo.fit(y)
    with pytest.raises(RuntimeError, match="fit"):
        modelo.predict(2)


@pytest.mark.parametrize(
    "kw",
    [
        {"lags": 0},
        {"lags": 2.5},
        {"capas": 1},
        {"unidades": 0},
        {"epochs": 0},
        {"batch_size": 0},
        {"learning_rate": 0},
        {"learning_rate": np.nan},
        {"dropout": 1},
        {"dropout": -0.1},
        {"l2": -1},
        {"l2": np.inf},
    ],
)
def test_configuracion_invalida(kw):
    with pytest.raises(ValueError):
        MLPForecaster(**kw)


def test_prediccion_exige_ajuste_y_horizonte_entero_positivo():
    modelo = MLPForecaster()
    with pytest.raises(RuntimeError):
        modelo.predict(1)
    modelo.fit(_serie())
    for h in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            modelo.predict(h)


def test_no_negatividad_es_configurable():
    y = np.full(30, -8.0)
    np.testing.assert_array_equal(MLPForecaster().fit(y).predict(3), 0.0)
    np.testing.assert_allclose(
        MLPForecaster(forzar_no_negativo=False).fit(y).predict(3), -8.0
    )


def test_entrada_extrema_falla_como_entrenamiento_no_finito():
    with pytest.raises(ValueError, match="entrenamiento DL no finito"):
        MLPForecaster().fit(np.full(30, 1e308))
