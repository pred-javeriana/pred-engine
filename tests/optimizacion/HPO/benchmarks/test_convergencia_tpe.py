"""TASK-HPO-4.0-C1: TPE debe converger mas rapido que busqueda aleatoria en
al menos 3 funciones objetivo (sphere, Rastrigin, Rosenbrock) -- las 3 que
exige Notion, por eso el criterio se prueba individualmente por funcion en
vez de con un conteo agregado."""

from __future__ import annotations

import pytest

from pred_engine.optimizacion.optimizadores.HPO.benchmarks.comparador_convergencia import (  # noqa: E501
    comparar_convergencia,
)
from pred_engine.optimizacion.optimizadores.HPO.benchmarks.funciones_objetivo import (
    rastrigin,
    rosenbrock,
    sphere,
)

_DIM = 3
_N_TRIALS = 60


@pytest.mark.slow
@pytest.mark.parametrize(
    ("nombre_funcion", "funcion", "bajo", "alto"),
    [
        ("sphere", sphere, -5.12, 5.12),
        ("rastrigin", rastrigin, -5.12, 5.12),
        ("rosenbrock", rosenbrock, -2.048, 2.048),
    ],
)
def test_tpe_converge_mas_cerca_del_minimo_global_que_random_search(
    nombre_funcion, funcion, bajo, alto
):
    resultado = comparar_convergencia(
        funcion,
        nombre_funcion=nombre_funcion,
        dim=_DIM,
        bajo=bajo,
        alto=alto,
        n_trials=_N_TRIALS,
        seed=0,
    )
    assert resultado.tpe_converge_mejor, (
        f"{nombre_funcion}: TPE={resultado.mejor_tpe:.4g} no supero a "
        f"random={resultado.mejor_aleatorio:.4g}"
    )
