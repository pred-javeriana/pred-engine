"""Casos conocidos y trampas deliberadas de las metricas de error."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pred_engine.comun.walkforward.metricas import (
    agregar,
    mae,
    mase,
    rmse,
    smape,
)


def test_mae_valor_conocido():
    assert mae(np.array([10.0, 20.0]), np.array([12.0, 18.0])) == pytest.approx(2.0)


def test_rmse_valor_conocido():
    assert rmse(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(
        math.sqrt((9 + 16) / 2)
    )


def test_smape_ceros_no_produce_nan_convencion_cero():
    assert smape(np.array([0.0]), np.array([0.0])) == 0.0


def test_smape_acotado_entre_0_y_200():
    valor = smape(np.array([1.0, 5.0, 0.0]), np.array([100.0, 0.0, 3.0]))
    assert 0.0 <= valor <= 200.0


def test_smape_prediccion_perfecta_es_cero():
    y = np.array([3.0, 5.0, 8.0])
    assert smape(y, y) == pytest.approx(0.0)


def test_mase_denominador_usa_solo_el_train_de_la_ventana():
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    valor = mase(np.array([6.0]), np.array([6.0]), y_train=y_train, estacionalidad=1)
    assert valor == pytest.approx(0.0)


def test_mase_infinito_si_train_constante_nunca_cero():
    y_train = np.full(10, 5.0)
    valor = mase(np.array([5.0]), np.array([7.0]), y_train=y_train, estacionalidad=1)
    assert math.isinf(valor)


def test_mase_prediccion_perfecta_con_train_no_degenerado_es_cero_no_infinito():
    y_train = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    valor = mase(np.array([4.0]), np.array([4.0]), y_train=y_train, estacionalidad=1)
    assert valor == pytest.approx(0.0)


def test_mase_train_insuficiente_para_estacionalidad_es_infinito():
    y_train = np.array([1.0, 2.0])
    valor = mase(np.array([3.0]), np.array([3.0]), y_train=y_train, estacionalidad=7)
    assert math.isinf(valor)


def test_agregar_media_recortada_ignora_outlier():
    valores = [1.0, 1.1, 0.9, 1.0, 100.0]
    resultado = agregar(valores, estrategia="media_recortada", proporcion_recorte=0.2)
    assert resultado < 2.0


def test_agregar_filtra_valores_no_finitos():
    valores = [1.0, 2.0, float("inf")]
    assert agregar(valores, estrategia="media") == pytest.approx(1.5)


def test_agregar_todo_no_finito_da_infinito():
    assert math.isinf(agregar([float("inf"), float("inf")], estrategia="media"))


def test_agregar_vacio_lanza_error():
    with pytest.raises(ValueError):
        agregar([])


def test_agregar_mediana():
    assert agregar([1.0, 2.0, 3.0], estrategia="mediana") == pytest.approx(2.0)


def test_agregar_estrategia_invalida():
    with pytest.raises(ValueError):
        agregar([1.0], estrategia="no_existe")  # type: ignore[arg-type]
