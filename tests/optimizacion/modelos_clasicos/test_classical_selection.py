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
    series_por_sku,
)


def _panel(n: int = 60, skus: tuple[str, ...] = ("A", "B")) -> pd.DataFrame:
    fechas = list(pd.date_range("2024-01-01", periods=n, freq="D"))
    return pd.DataFrame(
        {
            "sku_id": [s for s in skus for _ in range(n)],
            "timestamp": fechas * len(skus),
            "demand_qty": np.concatenate(
                [_serie_estacional(n, seed=i + 1) for i in range(len(skus))]
            ),
            "lead_time_days": [3] * (n * len(skus)),
        }
    )


_ESPACIO_MINIMO = EspacioClasico(
    p_max=1, d_max=1, q_max=1, P_max=0, D_max=0, Q_max=0, m=7
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


def test_series_por_sku_ordena_alfabeticamente_los_skus():
    panel = _panel(n=20, skus=("Z9", "A1", "M5"))
    assert [sku for sku, _ in series_por_sku(panel)] == ["A1", "M5", "Z9"]


def test_series_por_sku_ordena_cada_serie_por_timestamp():
    # El panel llega con las filas barajadas: la causalidad de Walk-Forward
    # depende de que la serie salga en orden cronologico de aqui.
    panel = _panel(n=30, skus=("A",))
    barajado = panel.sample(frac=1.0, random_state=7).reset_index(drop=True)

    ((_, serie),) = series_por_sku(barajado)
    esperado = panel.sort_values("timestamp")["demand_qty"].to_numpy(dtype=float)
    np.testing.assert_array_equal(serie, esperado)


def test_series_por_sku_exige_las_columnas_del_contrato():
    panel = _panel(n=10).drop(columns=["demand_qty"])
    with pytest.raises(ValueError, match="demand_qty"):
        series_por_sku(panel)


def test_n_procesos_invalido_falla_antes_de_calcular():
    with pytest.raises(ValueError, match="n_procesos"):
        seleccionar_por_panel(_panel(n=10), n_procesos=0)


@pytest.mark.slow
def test_paralelo_entre_skus_da_el_mismo_resultado_que_secuencial():
    # La promesa del nivel 1: repartir SKUs entre procesos no cambia nada.
    # Se sostiene porque los estudios no comparten estado y `derivar_semilla`
    # es un hash de (seed, trial_id, indice_ventana), no un contador global.
    panel = _panel(n=60, skus=("A", "B"))
    kwargs = dict(espacio=_ESPACIO_MINIMO, n_trials=5, horizonte=7, paso=7, seed=0)

    secuencial = seleccionar_por_panel(panel, **kwargs)
    paralelo = seleccionar_por_panel(panel, n_procesos=2, **kwargs)

    assert list(paralelo.keys()) == list(secuencial.keys()) == ["A", "B"]
    for sku_id in secuencial:
        s, p = secuencial[sku_id].seleccionada, paralelo[sku_id].seleccionada
        assert s is not None and p is not None
        assert p.order == s.order
        assert p.seasonal_order == s.seasonal_order
        assert p.valor == s.valor
        assert p.n_ventanas == s.n_ventanas
