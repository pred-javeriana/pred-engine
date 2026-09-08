"""`ejecutar_estudio`: el punto de entrada unico del HPO, extremo a extremo."""

from __future__ import annotations

import math

import numpy as np
from tests._dobles import (
    fabrica_falla_primeras_n,
    fabrica_nivel_constante,
    fabrica_rota,
)

from pred_engine.optimizacion.optimizadores.HPO.espacio import EspacioBusqueda, Flotante
from pred_engine.optimizacion.optimizadores.HPO.estudio import ejecutar_estudio
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda


def _serie_objetivo(nivel: float = 50.0, n: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return nivel + rng.normal(0, 1.0, size=n)


def _espacio() -> EspacioBusqueda:
    return EspacioBusqueda(parametros=(Flotante("nivel", 0.0, 100.0),))


def test_estudio_encuentra_el_optimo_conocido():
    y = _serie_objetivo(nivel=50.0)
    resultado = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_nivel_constante,
        n_trials=60,
        min_train=20,
        horizonte=1,
        paso=1,
        estacionalidad=1,
        seed=0,
        familia="prueba",
    )
    assert resultado.mejor is not None
    assert abs(resultado.mejor.configuracion["nivel"] - 50.0) < 5.0


def test_todos_los_trials_comparten_las_mismas_ventanas():
    y = _serie_objetivo()
    resultado = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_nivel_constante,
        n_trials=10,
        min_train=20,
        estacionalidad=1,
        seed=1,
    )
    assert len(resultado.ventanas) == len(y) - 20
    assert all(t.n_ventanas <= len(resultado.ventanas) for t in resultado.trials)


def test_determinismo_extremo_a_extremo():
    y = _serie_objetivo()
    r1 = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_nivel_constante,
        n_trials=15,
        min_train=20,
        estacionalidad=1,
        seed=42,
    )
    r2 = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_nivel_constante,
        n_trials=15,
        min_train=20,
        estacionalidad=1,
        seed=42,
    )
    assert r1.mejor is not None and r2.mejor is not None
    assert r1.mejor.configuracion == r2.mejor.configuracion
    assert r1.mejor.valor == r2.mejor.valor


def test_mejor_es_none_cuando_todo_falla():
    y = _serie_objetivo()
    resultado = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_rota,
        n_trials=5,
        min_train=20,
        estacionalidad=1,
        seed=0,
    )
    assert resultado.mejor is None
    assert resultado.n_completados == 0
    assert resultado.n_fallidos == 5


def test_fallo_en_las_primeras_ventanas_no_mata_el_trial_si_luego_converge():
    y = _serie_objetivo(n=60)
    resultado = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_falla_primeras_n(3),
        n_trials=1,
        min_train=20,
        horizonte=1,
        paso=1,
        estacionalidad=1,
        seed=0,
        familia="prueba",
    )
    trial = resultado.trials[0]
    assert trial.estado == "completado"
    assert trial.valor is not None
    assert math.isfinite(trial.valor)
    assert resultado.n_fallidos == 0


def test_fallo_en_todas_las_ventanas_hasta_la_poda_se_marca_fallido_no_podado():
    y = _serie_objetivo()
    resultado = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_rota,
        n_trials=3,
        min_train=20,
        estacionalidad=1,
        seed=0,
    )
    assert resultado.n_podados == 0
    assert resultado.n_fallidos == 3
    assert all(t.estado == "fallido" for t in resultado.trials)


def test_poda_reduce_las_ventanas_evaluadas_respecto_al_modo_sin_poda():
    y = _serie_objetivo()
    reglas_con_poda = ReglasPoda(min_ventanas=4, factor_reduccion=2)
    resultado = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_nivel_constante,
        n_trials=30,
        min_train=20,
        estacionalidad=1,
        seed=7,
        reglas=reglas_con_poda,
    )
    assert resultado.n_podados > 0
    ventanas_evaluadas = sum(t.n_ventanas for t in resultado.trials)
    ventanas_sin_poda = len(resultado.ventanas) * len(resultado.trials)
    assert ventanas_evaluadas < ventanas_sin_poda
