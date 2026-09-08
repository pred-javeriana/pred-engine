"""Componente 5: la regla mas importante es 'nunca podar antes de min_ventanas'.

`PodadorASHA` es un `optuna.pruners.BasePruner` real (ADR-02-006): las
pruebas construyen un `optuna.Study` de verdad y llaman `trial.should_prune()`,
en vez de invocar metodos propios (`registrar`/`decidir`) que ya no existen.
"""

from __future__ import annotations

import optuna
import pytest

from pred_engine.optimizacion.optimizadores.HPO.asha import PodadorASHA
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda


def _estudio(pruner: PodadorASHA) -> optuna.study.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=pruner,
    )


def _reportar(
    study: optuna.study.Study, valor: float, escalon: int
) -> optuna.trial.Trial:
    trial = study.ask()
    trial.report(valor, step=escalon)
    return trial


def test_ningun_trial_se_poda_antes_de_min_ventanas():
    pruner = PodadorASHA(ReglasPoda(min_ventanas=4), n_ventanas_totales=30)
    study = _estudio(pruner)
    trial = study.ask()
    for n in range(1, 4):
        trial.report(1000.0, step=n)
        assert trial.should_prune() is False


def test_n_ventanas_totales_menor_a_min_ventanas_es_invalido():
    with pytest.raises(ValueError):
        PodadorASHA(ReglasPoda(min_ventanas=4), n_ventanas_totales=2)


def test_escalones_arrancan_en_min_ventanas_y_terminan_en_el_total():
    pruner = PodadorASHA(
        ReglasPoda(min_ventanas=4, factor_reduccion=3), n_ventanas_totales=30
    )
    assert pruner.escalones()[0] == 4
    assert pruner.escalones()[-1] == 30
    assert list(pruner.escalones()) == sorted(pruner.escalones())


def test_promocion_del_mejor_tercio_en_un_escalon():
    pruner = PodadorASHA(
        ReglasPoda(min_ventanas=4, factor_reduccion=3), n_ventanas_totales=12
    )
    study = _estudio(pruner)
    escalon = pruner.escalones()[0]
    trials = {i: _reportar(study, float(i), escalon) for i in range(6)}
    assert trials[0].should_prune() is False
    assert trials[5].should_prune() is True


def test_motivo_de_poda_incluye_escalon_y_valores():
    pruner = PodadorASHA(
        ReglasPoda(min_ventanas=4, factor_reduccion=3), n_ventanas_totales=12
    )
    study = _estudio(pruner)
    escalon = pruner.escalones()[0]
    valores = {"a": 1.0, "b": 2.0, "c": 100.0}
    trials = {
        nombre: _reportar(study, valor, escalon) for nombre, valor in valores.items()
    }

    assert trials["c"].should_prune() is True
    assert pruner.ultimo_motivo is not None
    assert f"escalon={escalon}" in pruner.ultimo_motivo


def test_asincronia_orden_de_llegada_no_cambia_el_veredicto():
    reglas = ReglasPoda(min_ventanas=4, factor_reduccion=3)
    valores = {"t0": 1.0, "t1": 5.0, "t2": 9.0}

    pruner_a = PodadorASHA(reglas, n_ventanas_totales=12)
    study_a = _estudio(pruner_a)
    escalon = pruner_a.escalones()[0]
    trials_a = {
        nombre: _reportar(study_a, valores[nombre], escalon)
        for nombre in ("t0", "t1", "t2")
    }
    veredicto_a = {nombre: t.should_prune() for nombre, t in trials_a.items()}

    pruner_b = PodadorASHA(reglas, n_ventanas_totales=12)
    study_b = _estudio(pruner_b)
    trials_b = {
        nombre: _reportar(study_b, valores[nombre], escalon)
        for nombre in ("t2", "t0", "t1")
    }
    veredicto_b = {nombre: t.should_prune() for nombre, t in trials_b.items()}

    assert veredicto_a == veredicto_b


def test_sin_suficientes_pares_no_decide_todavia():
    pruner = PodadorASHA(
        ReglasPoda(min_ventanas=4, factor_reduccion=3), n_ventanas_totales=12
    )
    study = _estudio(pruner)
    escalon = pruner.escalones()[0]
    trial = _reportar(study, 1000.0, escalon)
    assert trial.should_prune() is False
