"""SarimaForecaster: contrato BaseForecaster + no-negatividad + fallos tipados."""

from __future__ import annotations

import numpy as np
import pytest
from tests._series import serie_ar_estacional

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
    modelo = SarimaForecaster(
        order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), tendencia="c"
    )
    y = _serie_ar1()
    assert modelo.fit(y) is modelo


def test_predict_longitud_igual_al_horizonte():
    modelo = SarimaForecaster(
        order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), tendencia="c"
    ).fit(_serie_ar1())
    pronostico = modelo.predict(5)
    assert len(pronostico) == 5


def test_predict_antes_de_fit_lanza_runtime_error():
    modelo = SarimaForecaster(
        order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), tendencia="c"
    )
    with pytest.raises(RuntimeError, match="fit"):
        modelo.predict(3)


def test_forzar_no_negativo_recorta_predicciones():
    modelo = SarimaForecaster(
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        tendencia="c",
        forzar_no_negativo=True,
    )
    y = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.0] * 10)
    modelo.fit(y)
    pronostico = modelo.predict(10)
    assert np.all(pronostico >= 0.0)


def test_determinismo_con_la_misma_configuracion():
    y = _serie_ar1()
    m1 = SarimaForecaster(
        order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), tendencia="c", seed=1
    ).fit(y)
    m2 = SarimaForecaster(
        order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), tendencia="c", seed=1
    ).fit(y)
    np.testing.assert_array_almost_equal(m1.predict(5), m2.predict(5))


def test_fit_con_serie_2d_lanza_value_error():
    modelo = SarimaForecaster(
        order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), tendencia="c"
    )
    with pytest.raises(ValueError, match="1D"):
        modelo.fit(np.ones((3, 3)))


def test_predict_con_horizonte_cero_lanza_value_error():
    modelo = SarimaForecaster(
        order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), tendencia="c"
    ).fit(_serie_ar1())
    with pytest.raises(ValueError, match="horizon"):
        modelo.predict(0)


def test_fallo_de_convergencia_se_convierte_en_ajuste_modelo_error(monkeypatch):
    import statsmodels.tsa.statespace.sarimax as sarimax_module

    def _fit_roto(self, *args, **kwargs):
        raise np.linalg.LinAlgError("matriz singular simulada")

    monkeypatch.setattr(sarimax_module.SARIMAX, "fit", _fit_roto)

    modelo = SarimaForecaster(
        order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), tendencia="c"
    )
    with pytest.raises(AjusteModeloError, match="no convergio"):
        modelo.fit(_serie_ar1())


def test_ajuste_en_el_borde_de_la_region_admisible_se_rechaza():
    # Prefijo walk-forward real del panel Kaggle (SKU 104::syn000, ventana 12):
    # 124 dias con 11 demandas. Sin restringir la estimacion el ajuste dejaba
    # raices AR dentro del circulo unitario y pronosticaba ~1e63; restringido,
    # una raiz MA queda a 1.006 del origen y el ajuste se rechaza (ADR-020).
    y = np.zeros(124)
    demandas = {4: 281, 10: 400, 19: 224, 46: 62, 59: 218, 76: 62}
    demandas |= {89: 219, 95: 1, 96: 416, 101: 448, 121: 35}
    for dia, valor in demandas.items():
        y[dia] = valor
    modelo = SarimaForecaster(
        order=(2, 0, 2), seasonal_order=(2, 0, 0, 7), tendencia="c"
    )

    with pytest.raises(AjusteModeloError, match="1.01"):
        modelo.fit(y)
    with pytest.raises(RuntimeError, match="fit"):
        modelo.predict(7)


def test_sobrediferenciar_ruido_blanco_deja_una_raiz_ma_unitaria_y_se_rechaza():
    ruido = 50 + np.random.default_rng(0).normal(0, 1, 150)
    with pytest.raises(AjusteModeloError, match="borde"):
        SarimaForecaster(order=(0, 2, 1), seasonal_order=(0, 0, 0, 0)).fit(ruido)


def test_un_ajuste_fallido_invalida_el_modelo_anterior():
    modelo = SarimaForecaster(
        order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), tendencia="c"
    )
    modelo.fit(_serie_ar1())
    # Una tendencia lineal sin diferenciar empuja la raiz AR al circulo unitario.
    tendencia = np.arange(150.0) + np.random.default_rng(0).normal(0, 1, 150)
    with pytest.raises(AjusteModeloError, match="borde"):
        modelo.fit(tendencia)
    with pytest.raises(RuntimeError, match="fit"):
        modelo.predict(3)


def test_pronosticar_filtra_lo_observado_con_los_parametros_del_ajuste():
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = serie_ar_estacional(90)
    historia, observadas = y[:70], y[70:]
    modelo = SarimaForecaster(
        order=(1, 0, 0), seasonal_order=(1, 0, 0, 7), tendencia="c"
    ).fit(historia)
    parametros = modelo._resultado_ajuste.params.copy()

    pronostico = modelo.pronosticar(observadas, 5)

    # Referencia: los mismos parametros filtrados sobre toda la serie observada.
    referencia = (
        SARIMAX(
            y,
            order=(1, 0, 0),
            seasonal_order=(1, 0, 0, 7),
            trend="c",
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        .filter(parametros)
        .forecast(5)
    )
    np.testing.assert_allclose(pronostico, np.clip(referencia, 0, None))
    np.testing.assert_array_equal(modelo._resultado_ajuste.params, parametros)


def _serie_semanal(n: int, ruido: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 20 + 5 * np.sin(np.arange(n) * 2 * np.pi / 7) + rng.normal(0, ruido, n)


def test_una_raiz_ar_estacional_en_el_borde_se_rechaza():
    # Un patron semanal casi exacto lleva el AR estacional a una raiz de modulo
    # 1.0002: la regla de ADR-020 cubre tambien los factores estacionales.
    modelo = SarimaForecaster(
        order=(0, 0, 0), seasonal_order=(1, 0, 0, 7), tendencia="c", seed=17
    )
    with pytest.raises(AjusteModeloError, match="1.01"):
        modelo.fit(_serie_semanal(56, ruido=0.02, seed=17))
    with pytest.raises(RuntimeError, match="fit"):
        modelo.predict(7)


def test_una_raiz_ma_estacional_en_el_borde_se_rechaza():
    # Diferenciar estacionalmente ruido blanco deja el MA estacional en el
    # circulo unitario.
    ruido = 50 + np.random.default_rng(0).normal(0, 1, 150)
    with pytest.raises(AjusteModeloError, match="borde"):
        SarimaForecaster(order=(0, 0, 0), seasonal_order=(0, 1, 1, 7)).fit(ruido)


def test_un_ajuste_estacional_holgado_se_acepta():
    y = serie_ar_estacional(90)
    pronostico = (
        SarimaForecaster(order=(1, 0, 0), seasonal_order=(1, 0, 0, 7), tendencia="c")
        .fit(y)
        .predict(7)
    )
    assert pronostico.shape == (7,)
    assert np.all(np.isfinite(pronostico))


@pytest.mark.parametrize(
    "y",
    [np.array([1.0, np.nan] * 30), np.array([1.0, np.inf] * 30), np.full(56, np.nan)],
)
def test_una_serie_no_finita_se_rechaza(y):
    modelo = SarimaForecaster(
        order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), tendencia="c"
    )
    with pytest.raises(ValueError, match="no finitos"):
        modelo.fit(y)
    with pytest.raises(RuntimeError, match="fit"):
        modelo.predict(3)


@pytest.mark.parametrize("y", [np.array([]), np.ones((10, 6)), np.full(60, np.nan)])
def test_un_reajuste_con_entrada_invalida_invalida_el_modelo_anterior(y):
    modelo = SarimaForecaster(
        order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), tendencia="c"
    ).fit(_serie_ar1())
    assert np.all(np.isfinite(modelo.predict(3)))
    with pytest.raises((ValueError, AjusteModeloError)):
        modelo.fit(y)
    with pytest.raises(RuntimeError, match="fit"):
        modelo.predict(3)
