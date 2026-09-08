"""Componente 8.2 (greedy): la prueba de equivalencia es la mas importante.

Si `iterar_walk_forward` agotado no da lo mismo que el modo
completo, el modo greedy y el completo dejan de ser comparables y todo el
benchmarking de ASHA pierde validez.
"""

from __future__ import annotations

import numpy as np
from tests._dobles import fabrica_espia, fabrica_ultimo_valor

from pred_engine.comun.walkforward.ventanas import generar_ventanas
from pred_engine.comun.walkforward.walk_forward import evaluar_walk_forward
from pred_engine.comun.walkforward.walk_forward_greedy import (
    EjecutorGreedy,
    iterar_walk_forward,
)


def _serie(n: int = 40, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(5, 15, size=n)


def test_greedy_agotado_es_identico_al_integro():
    y = _serie()
    ventanas = generar_ventanas(len(y), min_train=20, horizonte=1, paso=1)

    integro = evaluar_walk_forward(
        y,
        fabrica_ultimo_valor,
        {},
        min_train=20,
        horizonte=1,
        paso=1,
        seed=0,
        estacionalidad=1,
    )
    parciales = list(
        iterar_walk_forward(
            y, fabrica_ultimo_valor, {}, ventanas=ventanas, seed=0, estacionalidad=1
        )
    )

    assert parciales[-1].n_evaluadas == integro.n_ventanas_totales
    assert parciales[-1].valor_parcial == integro.valor_agregado
    for parcial, ventana_integra in zip(parciales, integro.ventanas, strict=True):
        np.testing.assert_array_equal(parcial.ultima.y_pred, ventana_integra.y_pred)
        np.testing.assert_array_equal(parcial.ultima.y_real, ventana_integra.y_real)
        assert parcial.ultima.metricas == ventana_integra.metricas


def test_ejecutor_greedy_via_resultado_tambien_es_identico_al_integro():
    y = _serie()
    ventanas = generar_ventanas(len(y), min_train=20, horizonte=1, paso=1)
    integro = evaluar_walk_forward(
        y,
        fabrica_ultimo_valor,
        {},
        min_train=20,
        horizonte=1,
        paso=1,
        seed=5,
        estacionalidad=1,
    )
    ejecutor = EjecutorGreedy(
        y, fabrica_ultimo_valor, {}, ventanas=ventanas, seed=5, estacionalidad=1
    )
    while ejecutor.avanzar() is not None:
        pass
    resultado = ejecutor.resultado()
    assert resultado.completo is True
    assert resultado.valor_agregado == integro.valor_agregado
    assert resultado.n_ventanas_evaluadas == integro.n_ventanas_evaluadas


def test_ejecutor_greedy_cierre_anticipado_marca_incompleto():
    y = _serie()
    ventanas = generar_ventanas(len(y), min_train=20, horizonte=1)
    ejecutor = EjecutorGreedy(
        y, fabrica_ultimo_valor, {}, ventanas=ventanas, seed=0, estacionalidad=1
    )
    ejecutor.avanzar()
    ejecutor.avanzar()
    ejecutor.cerrar("poda_de_prueba")
    resultado = ejecutor.resultado()
    assert resultado.completo is False
    assert resultado.n_ventanas_evaluadas == 2
    assert ejecutor.avanzar() is None
    assert ejecutor.motivo_cierre == "poda_de_prueba"


def test_ejecutor_greedy_agotado_marca_completo():
    y = _serie()
    ventanas = generar_ventanas(len(y), min_train=20, horizonte=1)
    ejecutor = EjecutorGreedy(
        y, fabrica_ultimo_valor, {}, ventanas=ventanas, seed=0, estacionalidad=1
    )
    while ejecutor.avanzar() is not None:
        pass
    resultado = ejecutor.resultado()
    assert resultado.completo is True
    assert resultado.n_ventanas_evaluadas == len(ventanas)
    assert ejecutor.motivo_cierre is None


def test_iterar_walk_forward_no_ve_datos_futuros():
    y = _serie()
    ventanas = generar_ventanas(len(y), min_train=20, horizonte=1)
    registro: list[int] = []
    for estado in iterar_walk_forward(
        y, fabrica_espia(registro), {}, ventanas=ventanas, estacionalidad=1
    ):
        assert registro[-1] == estado.ultima.ventana.n_train


def test_iterar_walk_forward_rechaza_ventanas_vacias():
    import pytest

    y = _serie()
    with pytest.raises(ValueError, match="ventanas"):
        list(iterar_walk_forward(y, fabrica_ultimo_valor, {}, ventanas=()))
