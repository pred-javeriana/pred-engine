"""Invariantes de `generar_ventanas`: la particion causal unica del proyecto."""

from __future__ import annotations

import pytest

from pred_engine.comun.walkforward.errores import VentanaInsuficienteError
from pred_engine.comun.walkforward.ventanas import generar_ventanas


def test_particion_causal_ejemplo_de_referencia():
    ventanas = generar_ventanas(9, min_train=5, horizonte=1, paso=1)
    assert len(ventanas) == 4
    esperado = [(0, 5, 5, 6), (0, 6, 6, 7), (0, 7, 7, 8), (0, 8, 8, 9)]
    obtenido = [
        (v.inicio_train, v.fin_train, v.inicio_val, v.fin_val) for v in ventanas
    ]
    assert obtenido == esperado


def test_sin_hueco_ni_solape():
    ventanas = generar_ventanas(30, min_train=10, horizonte=3, paso=2)
    assert len(ventanas) > 0
    for v in ventanas:
        assert v.inicio_val == v.fin_train


def test_fin_train_estrictamente_creciente():
    ventanas = generar_ventanas(30, min_train=10, horizonte=1, paso=1)
    fines = [v.fin_train for v in ventanas]
    assert fines == sorted(fines)
    assert len(set(fines)) == len(fines)


def test_ninguna_ventana_excede_n_observaciones():
    n = 25
    ventanas = generar_ventanas(n, min_train=10, horizonte=4, paso=3)
    assert len(ventanas) > 0
    assert all(v.fin_val <= n for v in ventanas)


def test_serie_insuficiente_lanza_error_con_deficit_exacto():
    with pytest.raises(VentanaInsuficienteError) as exc_info:
        generar_ventanas(4, min_train=5, horizonte=1)
    assert exc_info.value.requerido == 6
    assert exc_info.value.disponible == 4


def test_horizonte_mayor_a_uno():
    ventanas = generar_ventanas(15, min_train=5, horizonte=3, paso=3)
    assert len(ventanas) > 0
    assert all(v.horizonte == 3 for v in ventanas)


def test_paso_mayor_a_uno_espacia_las_ventanas():
    ventanas = generar_ventanas(30, min_train=10, horizonte=1, paso=5)
    fines = [v.fin_train for v in ventanas]
    assert fines == [10, 15, 20, 25]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_train": 0},
        {"min_train": 5, "horizonte": 0},
        {"min_train": 5, "paso": 0},
    ],
)
def test_parametros_invalidos_lanzan_value_error(kwargs):
    with pytest.raises(ValueError):
        generar_ventanas(30, **kwargs)
