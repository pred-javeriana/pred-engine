"""SarimaForecaster: contrato BaseForecaster + no-negatividad + fallos tipados."""

from __future__ import annotations

import numpy as np
import pytest

from pred_engine.comun.modelos.modelos_clasicos.errores import AjusteModeloError
from pred_engine.comun.modelos.modelos_clasicos.sarima import SarimaForecaster


def _serie_ar1(n: int = 60, phi: float = 0.6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    y[0] = rng.normal(50, 1)
    for t in range(1, n):
        y[t] = 50 * (1 - phi) + phi * y[t - 1] + rng.normal(0, 1)
    return y


def test_fit_devuelve_self():
    modelo = SarimaForecaster(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))
    y = _serie_ar1()
    assert modelo.fit(y) is modelo


def test_predict_longitud_igual_al_horizonte():
    modelo = SarimaForecaster(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0)).fit(
        _serie_ar1()
    )
    pronostico = modelo.predict(5)
    assert len(pronostico) == 5


def test_predict_antes_de_fit_lanza_runtime_error():
    modelo = SarimaForecaster(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))
    with pytest.raises(RuntimeError, match="fit"):
        modelo.predict(3)


def test_forzar_no_negativo_recorta_predicciones():
    modelo = SarimaForecaster(
        order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), forzar_no_negativo=True
    )
    y = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.0] * 10)
    modelo.fit(y)
    pronostico = modelo.predict(10)
    assert np.all(pronostico >= 0.0)


def test_determinismo_con_la_misma_configuracion():
    y = _serie_ar1()
    m1 = SarimaForecaster(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), seed=1).fit(y)
    m2 = SarimaForecaster(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), seed=1).fit(y)
    np.testing.assert_array_almost_equal(m1.predict(5), m2.predict(5))


def test_fit_con_serie_2d_lanza_value_error():
    modelo = SarimaForecaster(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))
    with pytest.raises(ValueError, match="1D"):
        modelo.fit(np.ones((3, 3)))


def test_predict_con_horizonte_cero_lanza_value_error():
    modelo = SarimaForecaster(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0)).fit(
        _serie_ar1()
    )
    with pytest.raises(ValueError, match="horizon"):
        modelo.predict(0)


def test_fallo_de_convergencia_se_convierte_en_ajuste_modelo_error(monkeypatch):
    import statsmodels.tsa.statespace.sarimax as sarimax_module

    def _fit_roto(self, *args, **kwargs):
        raise np.linalg.LinAlgError("matriz singular simulada")

    monkeypatch.setattr(sarimax_module.SARIMAX, "fit", _fit_roto)

    modelo = SarimaForecaster(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))
    with pytest.raises(AjusteModeloError, match="no convergio"):
        modelo.fit(_serie_ar1())


def test_estimacion_estacionaria_no_produce_pronosticos_explosivos():
    # Prefijo walk-forward real del panel Kaggle (SKU 104::syn000, ventana 12):
    # 124 dias con 11 demandas. Sin restringir la estimacion, el optimizador no
    # convergia, dejaba raices AR dentro del circulo unitario y pronosticaba ~1e63.
    y = np.zeros(124)
    demandas = {4: 281, 10: 400, 19: 224, 46: 62, 59: 218, 76: 62}
    demandas |= {89: 219, 95: 1, 96: 416, 101: 448, 121: 35}
    for dia, valor in demandas.items():
        y[dia] = valor

    modelo = SarimaForecaster(
        order=(2, 0, 2), seasonal_order=(2, 0, 0, 7), tendencia="c"
    ).fit(y)
    pronostico = modelo.predict(7)

    assert np.isfinite(pronostico).all()
    assert pronostico.max() <= y.max()
