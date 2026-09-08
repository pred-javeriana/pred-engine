"""Componente 8.1 (modo integro): recorre todas las ventanas, sin fuga, determinista."""

from __future__ import annotations

import numpy as np
from tests._dobles import fabrica_espia, fabrica_rota, fabrica_ultimo_valor

from pred_engine.comun.walkforward.walk_forward import evaluar_walk_forward


def _serie(n: int = 40, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(5, 15, size=n)


def test_recorre_todas_las_ventanas_y_marca_completo():
    y = _serie()
    resultado = evaluar_walk_forward(
        y, fabrica_ultimo_valor, {}, min_train=20, horizonte=1, paso=1, estacionalidad=1
    )
    assert resultado.completo is True
    assert resultado.n_ventanas_evaluadas == resultado.n_ventanas_totales
    assert resultado.n_ventanas_totales == len(y) - 20


def test_no_hay_fuga_temporal_el_modelo_solo_ve_el_train_de_su_ventana():
    y = _serie()
    registro: list[int] = []
    resultado = evaluar_walk_forward(
        y,
        fabrica_espia(registro),
        {},
        min_train=20,
        horizonte=1,
        paso=1,
        estacionalidad=1,
    )
    esperado = [v.ventana.n_train for v in resultado.ventanas]
    assert registro == esperado


def test_determinismo_con_la_misma_semilla():
    y = _serie()
    r1 = evaluar_walk_forward(
        y, fabrica_ultimo_valor, {}, min_train=20, seed=3, estacionalidad=1
    )
    r2 = evaluar_walk_forward(
        y, fabrica_ultimo_valor, {}, min_train=20, seed=3, estacionalidad=1
    )
    assert r1.valor_agregado == r2.valor_agregado
    for a, b in zip(r1.ventanas, r2.ventanas, strict=True):
        np.testing.assert_array_equal(a.y_pred, b.y_pred)


def test_tolerar_fallos_no_propaga_la_excepcion():
    y = _serie()
    resultado = evaluar_walk_forward(
        y, fabrica_rota, {}, min_train=20, tolerar_fallos=True, estacionalidad=1
    )
    assert resultado.completo is True
    assert all(v.fallo is not None for v in resultado.ventanas)
    assert resultado.valor_agregado == float("inf")


def test_sin_tolerar_fallos_propaga_la_excepcion():
    import pytest

    y = _serie()
    with pytest.raises(RuntimeError, match="deliberadamente roto"):
        evaluar_walk_forward(
            y, fabrica_rota, {}, min_train=20, tolerar_fallos=False, estacionalidad=1
        )
