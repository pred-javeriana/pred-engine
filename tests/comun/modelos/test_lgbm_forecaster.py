"""LightGBMForecaster: contrato de Pronosticador, determinismo y ausencia de fuga."""

from __future__ import annotations

import numpy as np
import pytest

from pred_engine.comun.modelos.modelos_machine_learning import (
    LightGBMForecaster,
    SerieCortaError,
    fabrica_ml,
)
from pred_engine.comun.modelos.modelos_machine_learning.lgbm import (
    MUESTRAS_MINIMAS,
    construir_muestras,
)
from pred_engine.comun.walkforward.protocolos import Pronosticador


def _serie(n: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = 10 + 5 * np.sin(np.arange(n) * 2 * np.pi / 7)
    return np.maximum(0.0, base + rng.normal(0, 1, n))


def _modelo(**kw) -> LightGBMForecaster:
    kw.setdefault("n_estimators", 30)
    return LightGBMForecaster(**kw)


def test_satisface_el_contrato_pronosticador():
    assert isinstance(_modelo(), Pronosticador)


def test_fit_devuelve_self_y_predict_tiene_la_longitud_del_horizonte():
    modelo = _modelo(lags=7, estacionalidad=7)
    assert modelo.fit(_serie()) is modelo
    assert modelo.predict(14).shape == (14,)


def test_predict_antes_de_fit_falla():
    with pytest.raises(RuntimeError):
        _modelo().predict(3)


@pytest.mark.parametrize("horizonte", [0, -1])
def test_horizonte_invalido(horizonte):
    with pytest.raises(ValueError, match="horizon"):
        _modelo().fit(_serie()).predict(horizonte)


def test_mismo_seed_y_configuracion_dan_pronosticos_identicos():
    y = _serie()
    a = _modelo(seed=3, subsample=0.7, colsample_bytree=0.7).fit(y).predict(7)
    b = _modelo(seed=3, subsample=0.7, colsample_bytree=0.7).fit(y).predict(7)
    np.testing.assert_array_equal(a, b)


def test_pronostico_nunca_es_negativo_en_demanda_intermitente():
    rng = np.random.default_rng(1)
    y = np.where(rng.random(120) < 0.7, 0.0, rng.integers(1, 20, 120)).astype(float)
    pronostico = _modelo(lags=5).fit(y).predict(14)
    assert np.all(pronostico >= 0.0)


def test_serie_constante_se_pronostica_constante():
    pronostico = _modelo(lags=4).fit(np.full(60, 5.0)).predict(5)
    np.testing.assert_allclose(pronostico, 5.0, atol=1e-6)


def test_aprende_la_estacionalidad_semanal():
    y = _serie(n=200)
    pronostico = _modelo(lags=7, estacionalidad=7, n_estimators=80).fit(y).predict(7)
    esperado = y[-7:]  # ciclo semanal: la ultima semana es la mejor referencia
    assert np.mean(np.abs(pronostico - esperado)) < 3.0


def test_serie_mas_corta_que_lags_mas_muestras_minimas_falla_con_mensaje_claro():
    modelo = _modelo(lags=10)
    with pytest.raises(SerieCortaError, match="insuficiente"):
        modelo.fit(np.arange(10 + MUESTRAS_MINIMAS - 1, dtype=float))


def test_serie_en_el_limite_ajusta():
    _modelo(lags=10).fit(_serie(10 + MUESTRAS_MINIMAS))


def test_serie_no_finita_o_no_1d_se_rechaza():
    with pytest.raises(ValueError, match="no finitos"):
        _modelo().fit(np.array([1.0, np.nan] * 30))
    with pytest.raises(ValueError, match="1D"):
        _modelo().fit(np.ones((10, 2)))


@pytest.mark.parametrize("kw", [{"lags": 0}, {"estacionalidad": 0}])
def test_parametros_invalidos_en_el_constructor(kw):
    with pytest.raises(ValueError):
        LightGBMForecaster(**kw)


def test_las_muestras_no_filtran_el_objetivo_ni_el_futuro():
    serie = np.arange(30, dtype=float)
    lags = 5
    X, objetivos = construir_muestras(serie, lags, estacionalidad=1)
    assert len(X) == len(objetivos) == len(serie) - lags
    for i in range(len(X)):
        np.testing.assert_array_equal(X[i, :lags], serie[i : i + lags])
        assert objetivos[i] == serie[i + lags]
        # Todos los valores usados como feature son anteriores al objetivo.
        assert X[i, :lags].max() < objetivos[i]


def test_cambiar_el_futuro_no_altera_el_ajuste_previo():
    # `fit` solo ve el `y` que recibe: perturbar observaciones POSTERIORES al
    # corte no puede cambiar el modelo entrenado con y[:corte].
    y = _serie()
    corte = 100
    y_alterada = y.copy()
    y_alterada[corte:] += 1000.0
    a = _modelo(seed=1).fit(y[:corte]).predict(7)
    b = _modelo(seed=1).fit(y_alterada[:corte]).predict(7)
    np.testing.assert_array_equal(a, b)


def test_fabrica_ml_traduce_la_configuracion_y_usa_defaults():
    modelo = fabrica_ml(
        {"lags": 9, "max_depth": 3, "learning_rate": 0.05, "m": 7}, seed=4
    )
    assert isinstance(modelo, LightGBMForecaster)
    assert (modelo.lags, modelo.max_depth, modelo.estacionalidad) == (9, 3, 7)
    assert modelo.learning_rate == 0.05
    assert modelo.seed == 4
    assert fabrica_ml({}, seed=0).lags == 7  # default
