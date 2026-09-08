"""Contratos del motor de HPO: el backend real debe satisfacerlos.

Esta prueba es la que sostiene la promesa de desacoplamiento: si algun dia
`optuna.trial.Trial` deja de exponer estos metodos (o se cambia de backend
sin actualizar el adaptador), esta prueba es la primera en fallar -- antes
de que `estudio.py` se entere en produccion.
"""

from __future__ import annotations

import optuna

from pred_engine.optimizacion.optimizadores.HPO.contratos import InfoTrial, TrialHPO


def test_trial_de_optuna_satisface_trialhpo_estructuralmente():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    assert isinstance(trial, TrialHPO)


def test_info_trial_es_inmutable_y_guarda_lo_esperado():
    info = InfoTrial(
        numero=3,
        estado="completado",
        valor=1.5,
        parametros={"p": 1},
        atributos={"trial_id": "clasicos-SKU1-0003"},
    )
    assert info.numero == 3
    assert info.estado == "completado"
    assert info.valor == 1.5
    assert info.parametros == {"p": 1}
    assert info.atributos["trial_id"] == "clasicos-SKU1-0003"
