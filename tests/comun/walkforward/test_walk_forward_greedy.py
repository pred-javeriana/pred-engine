"""Componente 8.2 (greedy): la prueba de equivalencia es la mas importante.

Si `iterar_walk_forward` agotado no da lo mismo que el modo
completo, el modo greedy y el completo dejan de ser comparables y todo el
benchmarking de ASHA pierde validez.
"""

from __future__ import annotations

import numpy as np
import pytest
from tests._dobles import fabrica_espia, fabrica_falla_primeras_n, fabrica_ultimo_valor

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
    y = _serie()
    with pytest.raises(ValueError, match="ventanas"):
        list(iterar_walk_forward(y, fabrica_ultimo_valor, {}, ventanas=()))


def test_tolerar_fallos_es_la_unica_diferencia_real_entre_los_dos_modos():
    """Con los defaults, `evaluar_walk_forward` es estricto (relanza el
    fallo de ajuste) y `EjecutorGreedy` es tolerante (continua). Esto es
    intencional (ver el docstring de `walk_forward.py`): el greedy es el
    motor de BUSQUEDA del HPO, donde una ventana que no converge es señal
    para ASHA, no motivo para abortar todo el trial.

    La prueba de "agotado es identico al integro" de arriba usa una fabrica
    que nunca falla, asi que nunca ejercita esta diferencia. Esta prueba
    hace las dos cosas: (1) confirma que el integro por defecto SI relanza,
    y (2) confirma que, pidiendo tolerancia explicita en ambos modos, el
    agregado sigue siendo identico -- la equivalencia real esta en el
    algoritmo, no en el manejo de fallos.
    """
    y = _serie()
    ventanas = generar_ventanas(len(y), min_train=20, horizonte=1, paso=1)

    with pytest.raises(RuntimeError):
        evaluar_walk_forward(
            y,
            fabrica_falla_primeras_n(2),
            {},
            min_train=20,
            horizonte=1,
            paso=1,
            seed=0,
            estacionalidad=1,
        )  # tolerar_fallos=False por defecto -> relanza

    integro_tolerante = evaluar_walk_forward(
        y,
        fabrica_falla_primeras_n(2),
        {},
        min_train=20,
        horizonte=1,
        paso=1,
        seed=0,
        estacionalidad=1,
        tolerar_fallos=True,
    )

    ejecutor = EjecutorGreedy(
        y,
        fabrica_falla_primeras_n(2),
        {},
        ventanas=ventanas,
        seed=0,
        estacionalidad=1,
    )  # tolerar_fallos=True por defecto
    while ejecutor.avanzar() is not None:
        pass
    greedy_tolerante = ejecutor.resultado()

    assert greedy_tolerante.valor_agregado == integro_tolerante.valor_agregado
    assert (
        greedy_tolerante.n_ventanas_evaluadas == integro_tolerante.n_ventanas_evaluadas
    )
    assert greedy_tolerante.n_ventanas_totales == integro_tolerante.n_ventanas_totales


def test_proporcion_recorte_cambia_el_agregado_en_ambos_modos():
    """`proporcion_recorte` debe llegar de verdad hasta `valor_agregado_de`
    en los dos modos -- antes no se propagaba en el greedy (ni existia el
    parametro en el integro)."""
    y = _serie()
    ventanas = generar_ventanas(len(y), min_train=20, horizonte=1, paso=1)

    integro_sin_recorte = evaluar_walk_forward(
        y, fabrica_ultimo_valor, {}, min_train=20, horizonte=1, paso=1,
        seed=0, estacionalidad=1, agregacion="media_recortada", proporcion_recorte=0.0,
    )
    integro_con_recorte = evaluar_walk_forward(
        y, fabrica_ultimo_valor, {}, min_train=20, horizonte=1, paso=1,
        seed=0, estacionalidad=1, agregacion="media_recortada", proporcion_recorte=0.3,
    )

    ejecutor_sin_recorte = EjecutorGreedy(
        y, fabrica_ultimo_valor, {}, ventanas=ventanas, seed=0, estacionalidad=1,
        agregacion="media_recortada", proporcion_recorte=0.0,
    )
    while ejecutor_sin_recorte.avanzar() is not None:
        pass

    ejecutor_con_recorte = EjecutorGreedy(
        y, fabrica_ultimo_valor, {}, ventanas=ventanas, seed=0, estacionalidad=1,
        agregacion="media_recortada", proporcion_recorte=0.3,
    )
    while ejecutor_con_recorte.avanzar() is not None:
        pass

    # mismo backend (media_recortada), distinta proporcion -> agregados distintos
    assert integro_sin_recorte.valor_agregado != integro_con_recorte.valor_agregado
    assert (
        ejecutor_sin_recorte.resultado().valor_agregado
        != ejecutor_con_recorte.resultado().valor_agregado
    )
    # y el greedy coincide con el integro para cada proporcion (comparabilidad)
    assert (
        ejecutor_sin_recorte.resultado().valor_agregado
        == integro_sin_recorte.valor_agregado
    )
    assert (
        ejecutor_con_recorte.resultado().valor_agregado
        == integro_con_recorte.valor_agregado
    )
