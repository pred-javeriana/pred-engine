"""Componente 9 (parcial): 'podado' y 'fallido' deben ser distinguibles, con motivo.

`RegistroEstudio` ya no existe (ADR-02-006): el `optuna.Study` es la fuente
de verdad. Estas pruebas ejercitan las funciones puente
(`instantanea_desde_estudio`, `volcar_jsonl`, `reanudar_estudio`) contra un
`Study` real.
"""

from __future__ import annotations

from pathlib import Path

import optuna
from optuna.trial import TrialState

from pred_engine.optimizacion.optimizadores.HPO.asha import PodadorASHA
from pred_engine.optimizacion.optimizadores.HPO.espacio import Entero, EspacioBusqueda
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda
from pred_engine.optimizacion.optimizadores.HPO.registro import (
    instantanea_desde_estudio,
    reanudar_estudio,
    volcar_jsonl,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _ventanas():
    from pred_engine.comun.dataclasses.validacion_temporal import VentanaTemporal

    return (
        VentanaTemporal(
            indice=0, inicio_train=0, fin_train=10, inicio_val=10, fin_val=11
        ),
    )


def _espacio() -> EspacioBusqueda:
    return EspacioBusqueda(parametros=(Entero("p", 0, 5),))


def _estudio_vacio() -> optuna.study.Study:
    pruner = PodadorASHA(ReglasPoda(min_ventanas=1), n_ventanas_totales=1)
    return optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=pruner,
    )


def test_podado_y_fallido_son_estados_distintos_con_motivo_no_nulo():
    study = _estudio_vacio()

    t1 = study.ask()
    t1.suggest_int("p", 0, 5)
    t1.set_user_attr("trial_id", "t1")
    t1.set_user_attr("n_ventanas", 4)
    t1.set_user_attr("motivo", "ventana=4 valor=2.0 asha")
    t1.report(2.0, step=4)
    study.tell(t1, state=TrialState.PRUNED)

    t2 = study.ask()
    t2.suggest_int("p", 0, 5)
    t2.set_user_attr("trial_id", "t2")
    t2.set_user_attr("motivo", "excepcion_de_ajuste")
    study.tell(t2, state=TrialState.FAIL)

    resultado = instantanea_desde_estudio(
        study, ventanas=_ventanas(), metrica_objetivo="mase", seed=0
    )
    trials = {t.id: t for t in resultado.trials}
    assert trials["t1"].estado == "podado"
    assert trials["t1"].motivo is not None and "asha" in trials["t1"].motivo
    assert trials["t2"].estado == "fallido"
    assert trials["t2"].motivo == "excepcion_de_ajuste"
    assert trials["t1"].estado != trials["t2"].estado


def test_jsonl_round_trip(tmp_path: Path):
    study = _estudio_vacio()
    t1 = study.ask()
    t1.suggest_int("p", 0, 5)
    t1.set_user_attr("trial_id", "t1")
    t1.set_user_attr("n_ventanas", 10)
    study.tell(t1, 0.5, state=TrialState.COMPLETE)

    ruta = tmp_path / "estudio.jsonl"
    volcar_jsonl(study, ruta)

    lineas = ruta.read_text(encoding="utf-8").strip().splitlines()
    assert len(lineas) == 1
    assert lineas[0]

    pruner = PodadorASHA(ReglasPoda(min_ventanas=1), n_ventanas_totales=1)
    reanudado = reanudar_estudio(
        ruta,
        espacio=_espacio(),
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=pruner,
    )
    resultado = instantanea_desde_estudio(
        reanudado, ventanas=_ventanas(), metrica_objetivo="mase", seed=0
    )
    assert resultado.trials[0].id == "t1"
    assert resultado.trials[0].valor == 0.5


def test_reanudar_sin_archivo_devuelve_estudio_vacio(tmp_path: Path):
    pruner = PodadorASHA(ReglasPoda(min_ventanas=1), n_ventanas_totales=1)
    reanudado = reanudar_estudio(
        tmp_path / "no_existe.jsonl",
        espacio=_espacio(),
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=pruner,
    )
    assert reanudado.get_trials(deepcopy=False) == []


def test_instantanea_elige_el_menor_valor():
    study = _estudio_vacio()
    t1 = study.ask()
    t1.suggest_int("p", 0, 5)
    t1.set_user_attr("trial_id", "t1")
    study.tell(t1, 2.0, state=TrialState.COMPLETE)

    t2 = study.ask()
    t2.suggest_int("p", 0, 5)
    t2.set_user_attr("trial_id", "t2")
    study.tell(t2, 0.5, state=TrialState.COMPLETE)

    resultado = instantanea_desde_estudio(
        study, ventanas=_ventanas(), metrica_objetivo="mase", seed=0
    )
    assert resultado.mejor is not None
    assert resultado.mejor.id == "t2"
    assert resultado.n_completados == 2


def test_instantanea_mejor_es_none_sin_completados():
    study = _estudio_vacio()
    t1 = study.ask()
    t1.suggest_int("p", 0, 5)
    t1.set_user_attr("trial_id", "t1")
    t1.set_user_attr("motivo", "x")
    study.tell(t1, state=TrialState.FAIL)

    resultado = instantanea_desde_estudio(
        study, ventanas=_ventanas(), metrica_objetivo="mase", seed=0
    )
    assert resultado.mejor is None
    assert resultado.n_fallidos == 1
