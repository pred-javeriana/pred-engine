"""Muestreadores de HPO: como se propone la siguiente configuracion a evaluar."""

from __future__ import annotations

import optuna


def construir_muestreador_tpe(
    *,
    seed: int = 0,
    n_arranque: int = 10,
    n_candidatos: int = 24,
    multivariate: bool = True,
) -> optuna.samplers.TPESampler:
    return optuna.samplers.TPESampler(
        seed=seed,
        multivariate=multivariate,
        n_startup_trials=n_arranque,
        n_ei_candidates=n_candidatos,
    )


def construir_muestreador_aleatorio(*, seed: int = 0) -> optuna.samplers.RandomSampler:
    return optuna.samplers.RandomSampler(seed=seed)
