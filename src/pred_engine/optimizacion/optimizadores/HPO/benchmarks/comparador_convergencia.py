"""Compara la convergencia de TPE vs busqueda aleatoria (TASK-HPO-4.0-C1).

Las funciones de `funciones_objetivo.py` son deterministas y sin dependencia
temporal, asi que se optimizan directamente con el ciclo `ask`/`suggest_float`
/`tell` de Optuna -- el mismo mecanismo que `estudio.py` usa por debajo para
series de tiempo -- sin pasar por Walk-Forward.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import optuna

from pred_engine.optimizacion.optimizadores.HPO.muestreadores import (
    construir_muestreador_aleatorio,
    construir_muestreador_tpe,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True, slots=True)
class ResultadoConvergencia:
    nombre_funcion: str
    mejor_tpe: float
    mejor_aleatorio: float

    @property
    def tpe_converge_mejor(self) -> bool:
        return self.mejor_tpe < self.mejor_aleatorio


def _optimizar(
    funcion: Callable[[Mapping[str, float]], float],
    *,
    dim: int,
    bajo: float,
    alto: float,
    n_trials: int,
    muestreador: optuna.samplers.BaseSampler,
) -> float:
    study = optuna.create_study(direction="minimize", sampler=muestreador)
    for _ in range(n_trials):
        trial = study.ask()
        configuracion = {
            f"x{i}": trial.suggest_float(f"x{i}", bajo, alto) for i in range(dim)
        }
        study.tell(trial, funcion(configuracion))
    return study.best_value


def comparar_convergencia(
    funcion: Callable[[Mapping[str, float]], float],
    *,
    nombre_funcion: str,
    dim: int = 3,
    bajo: float = -5.12,
    alto: float = 5.12,
    n_trials: int = 60,
    seed: int = 0,
) -> ResultadoConvergencia:
    """Optimiza `funcion` con TPE y con busqueda aleatoria bajo el mismo
    presupuesto (`n_trials`) y semilla, y reporta el mejor valor de cada
    uno para comparar cual convergio mas cerca del minimo global (0.0)."""
    mejor_tpe = _optimizar(
        funcion,
        dim=dim,
        bajo=bajo,
        alto=alto,
        n_trials=n_trials,
        muestreador=construir_muestreador_tpe(seed=seed),
    )
    mejor_aleatorio = _optimizar(
        funcion,
        dim=dim,
        bajo=bajo,
        alto=alto,
        n_trials=n_trials,
        muestreador=construir_muestreador_aleatorio(seed=seed),
    )
    return ResultadoConvergencia(
        nombre_funcion=nombre_funcion,
        mejor_tpe=mejor_tpe,
        mejor_aleatorio=mejor_aleatorio,
    )
