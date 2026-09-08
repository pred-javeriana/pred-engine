"""Construccion de los samplers de Optuna usados por el motor de HPO (ADR-02-006)."""

from __future__ import annotations

import optuna

from pred_engine.optimizacion.optimizadores.HPO.muestreadores import (
    construir_muestreador_aleatorio,
    construir_muestreador_tpe,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def test_construir_muestreador_tpe_usa_multivariado_por_defecto():
    sampler = construir_muestreador_tpe(seed=0)
    assert isinstance(sampler, optuna.samplers.TPESampler)
    # `_multivariate` es un detalle interno de Optuna (no hay accessor
    # publico); se verifica igual porque `multivariate=True` es la decision
    # de diseno explicita de ADR-02-006.
    assert sampler._multivariate is True


def test_construir_muestreador_tpe_reproducible_con_misma_semilla():
    def _diez_sugerencias(seed: int) -> list[float]:
        sampler = construir_muestreador_tpe(seed=seed, n_arranque=2)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        valores = []
        for _ in range(10):
            trial = study.ask()
            x = trial.suggest_float("x", 0.0, 100.0)
            valores.append(x)
            study.tell(trial, abs(x - 50.0))
        return valores

    assert _diez_sugerencias(7) == _diez_sugerencias(7)


def test_construir_muestreador_aleatorio_cubre_el_rango():
    sampler = construir_muestreador_aleatorio(seed=1)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    valores = []
    for _ in range(200):
        trial = study.ask()
        valores.append(trial.suggest_float("x", 0.0, 100.0))
        study.tell(trial, 0.0)
    assert min(valores) < 20.0
    assert max(valores) > 80.0


def test_tpe_concentra_sugerencias_cerca_del_optimo_conocido():
    sampler = construir_muestreador_tpe(seed=1, n_arranque=8)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    for _ in range(40):
        trial = study.ask()
        x = trial.suggest_float("x", 0.0, 100.0)
        study.tell(trial, abs(x - 70.0))

    sugerencias = []
    for _ in range(30):
        trial = study.ask()
        sugerencias.append(trial.suggest_float("x", 0.0, 100.0))
        study.tell(trial, abs(trial.params["x"] - 70.0))
    error_tpe = sum(abs(v - 70.0) for v in sugerencias) / len(sugerencias)

    sampler_aleatorio = construir_muestreador_aleatorio(seed=1)
    study_aleatorio = optuna.create_study(
        direction="minimize", sampler=sampler_aleatorio
    )
    referencia = []
    for _ in range(30):
        trial = study_aleatorio.ask()
        referencia.append(trial.suggest_float("x", 0.0, 100.0))
        study_aleatorio.tell(trial, abs(trial.params["x"] - 70.0))
    error_aleatorio = sum(abs(v - 70.0) for v in referencia) / len(referencia)

    assert error_tpe < error_aleatorio
