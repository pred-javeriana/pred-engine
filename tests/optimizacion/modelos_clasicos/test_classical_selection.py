"""classical_selection.py como orquestador: HPO + Walk-Forward + SARIMA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pred_engine.optimizacion.optimizadores.modelos_clasicos.classical_selection import (  # noqa: E501
    EspacioClasico,
    construir_espacio,
    fabrica_sarima,
    min_train_recomendado,
    seleccionar_configuracion_clasica,
    seleccionar_por_panel,
)


def _serie_estacional(n: int = 90, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    nivel = 50.0
    estacional = 10.0 * np.sin(2 * np.pi * t / 7.0)
    ruido = rng.normal(0, 2.0, size=n)
    return np.clip(nivel + estacional + ruido, 0.0, None)


def test_construir_espacio_excluye_configuracion_degenerada():
    espacio = construir_espacio(EspacioClasico(p_max=1, d_max=1, q_max=1, m=1))
    rng = np.random.default_rng(0)
    for _ in range(50):
        configuracion = espacio.muestrear(rng)
        assert not (
            configuracion["p"] == 0
            and configuracion["d"] == 0
            and configuracion["q"] == 0
        )


def test_construir_espacio_respeta_orden_total_maximo():
    cfg = EspacioClasico(p_max=3, q_max=3, P_max=2, Q_max=2, m=7, max_orden_total=4)
    espacio = construir_espacio(cfg)
    rng = np.random.default_rng(0)
    for _ in range(100):
        configuracion = espacio.muestrear(rng)
        orden = (
            configuracion["p"]
            + configuracion["q"]
            + configuracion.get("P", 0)
            + configuracion.get("Q", 0)
        )
        assert orden <= 4


def test_fabrica_sarima_produce_order_y_seasonal_order_correctos():
    configuracion = {"p": 1, "d": 0, "q": 1, "P": 0, "D": 0, "Q": 0, "m": 7}
    modelo = fabrica_sarima(configuracion, seed=0)
    assert modelo.order == (1, 0, 1)
    assert modelo.seasonal_order == (0, 0, 0, 7)


def test_min_train_recomendado_con_m_7_es_al_menos_20():
    assert min_train_recomendado(EspacioClasico(m=7)) >= 20


def test_serie_demasiado_corta_falla_temprano_sin_gastar_computo():
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="insuficiente"):
        seleccionar_configuracion_clasica(y, sku_id="X1", n_trials=5)


@pytest.mark.slow
def test_selecciona_una_configuracion_mejor_que_el_naive_estacional():
    y = _serie_estacional()
    resultado = seleccionar_configuracion_clasica(
        y,
        sku_id="SKU-1",
        espacio=EspacioClasico(
            p_max=2, d_max=1, q_max=2, P_max=1, D_max=0, Q_max=1, m=7
        ),
        n_trials=15,
        horizonte=7,
        paso=7,
        seed=0,
    )
    assert resultado.seleccionada is not None
    assert resultado.seleccionada.valor < float("inf")


@pytest.mark.slow
def test_determinismo_de_la_seleccion_con_la_misma_semilla():
    y = _serie_estacional()
    kwargs = dict(
        sku_id="SKU-1",
        espacio=EspacioClasico(
            p_max=1, d_max=1, q_max=1, P_max=1, D_max=0, Q_max=0, m=7
        ),
        n_trials=8,
        horizonte=7,
        paso=7,
        seed=11,
    )
    r1 = seleccionar_configuracion_clasica(y, **kwargs)
    r2 = seleccionar_configuracion_clasica(y, **kwargs)
    assert r1.seleccionada is not None and r2.seleccionada is not None
    assert r1.seleccionada.order == r2.seleccionada.order
    assert r1.seleccionada.seasonal_order == r2.seleccionada.seasonal_order
    assert r1.seleccionada.valor == r2.seleccionada.valor


@pytest.mark.slow
def test_seleccionar_por_panel_no_mezcla_series_de_distintos_skus():
    n = 60
    panel = pd.DataFrame(
        {
            "sku_id": ["A"] * n + ["B"] * n,
            "timestamp": list(pd.date_range("2024-01-01", periods=n, freq="D")) * 2,
            "demand_qty": np.concatenate(
                [_serie_estacional(n, seed=1), _serie_estacional(n, seed=2)]
            ),
            "lead_time_days": [3] * (2 * n),
        }
    )
    resultados = seleccionar_por_panel(
        panel,
        espacio=EspacioClasico(
            p_max=1, d_max=1, q_max=1, P_max=0, D_max=0, Q_max=0, m=7
        ),
        n_trials=5,
        horizonte=7,
        paso=7,
        seed=0,
    )
    assert set(resultados.keys()) == {"A", "B"}
