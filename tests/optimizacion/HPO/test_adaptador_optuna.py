"""Adaptador Optuna: el unico modulo del motor de HPO que conoce Optuna.

El algoritmo puro de ASHA se prueba en `test_asha.py` (sin Optuna) y la
traduccion al contrato externo en `test_registro.py` (tampoco necesita
Optuna). Aqui solo se prueba que el adaptador conecta correctamente ambas
piezas con un `optuna.Study` real: `should_prune()` de verdad, el motivo de
poda indexado por trial (no compartido), la preservacion del valor en
trials podados, y la persistencia JSONL.
"""

from __future__ import annotations

from pathlib import Path

import optuna

from pred_engine.optimizacion.optimizadores.HPO.adaptador_optuna import (
    PodadorASHAOptuna,
    crear_estudio_optuna,
    reanudar_estudio,
    trials_desde_study,
    volcar_jsonl,
)
from pred_engine.optimizacion.optimizadores.HPO.asha import DecisorASHA
from pred_engine.optimizacion.optimizadores.HPO.espacio import Entero, EspacioBusqueda
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _decisor(
    min_ventanas: int = 1, n_ventanas_totales: int = 1, factor_reduccion: int = 3
) -> DecisorASHA:
    return DecisorASHA(
        ReglasPoda(min_ventanas=min_ventanas, factor_reduccion=factor_reduccion),
        n_ventanas_totales=n_ventanas_totales,
    )


def _espacio() -> EspacioBusqueda:
    return EspacioBusqueda(parametros=(Entero("p", 0, 5),))


def test_crear_estudio_optuna_ask_devuelve_un_trialhpo_utilizable():
    study, podador = crear_estudio_optuna(
        muestreador=optuna.samplers.RandomSampler(seed=0), decisor_asha=_decisor()
    )
    trial = study.ask()
    assert trial.number == 0
    assert podador.escalones() == (1,)


def test_tell_podado_toma_el_valor_del_ultimo_report():
    # Optuna PROHIBE pasar un valor explicito junto con PRUNED/FAIL
    # (ValueError) -- lo recupera solo, del ultimo `trial.report()`. Por eso
    # `_EstudioOptuna.tell` nunca pasa valor para "podado"/"fallido", y
    # `_correr_trial` siempre reporta antes de pedir el cierre.
    study, _ = crear_estudio_optuna(
        muestreador=optuna.samplers.RandomSampler(seed=0), decisor_asha=_decisor()
    )
    trial = study.ask()
    trial.suggest_int("p", 0, 5)
    trial.report(3.5, 4)
    study.tell(trial, None, estado="podado")

    (info,) = study.trials_finalizados()
    assert info.estado == "podado"
    assert info.valor == 3.5


def test_tell_podado_ignora_el_argumento_valor_y_usa_el_ultimo_report():
    # El adaptador nunca reenvia `valor` a Optuna para "podado"/"fallido"
    # (Optuna lo rechazaria con ValueError -- ver el docstring de `tell`).
    # Documenta que la fuente de verdad es SIEMPRE el ultimo `trial.report()`,
    # sin importar que argumento `valor` reciba `tell` para este estado.
    study, _ = crear_estudio_optuna(
        muestreador=optuna.samplers.RandomSampler(seed=0), decisor_asha=_decisor()
    )
    trial = study.ask()
    trial.suggest_int("p", 0, 5)
    trial.report(3.5, 4)
    study.tell(trial, 999.0, estado="podado")  # el 999.0 se ignora

    (info,) = study.trials_finalizados()
    assert info.valor == 3.5


def test_tell_fallido_sin_valor():
    study, _ = crear_estudio_optuna(
        muestreador=optuna.samplers.RandomSampler(seed=0), decisor_asha=_decisor()
    )
    trial = study.ask()
    trial.suggest_int("p", 0, 5)
    study.tell(trial, None, estado="fallido")

    (info,) = study.trials_finalizados()
    assert info.estado == "fallido"
    assert info.valor is None


def test_tell_completado_con_valor():
    study, _ = crear_estudio_optuna(
        muestreador=optuna.samplers.RandomSampler(seed=0), decisor_asha=_decisor()
    )
    trial = study.ask()
    trial.suggest_int("p", 0, 5)
    study.tell(trial, 1.25, estado="completado")

    (info,) = study.trials_finalizados()
    assert info.estado == "completado"
    assert info.valor == 1.25


def test_podador_asha_optuna_poda_de_verdad_con_should_prune():
    decisor = _decisor(min_ventanas=4, n_ventanas_totales=12, factor_reduccion=3)
    study, podador = crear_estudio_optuna(
        muestreador=optuna.samplers.RandomSampler(seed=0), decisor_asha=decisor
    )
    escalon = decisor.escalones()[0]
    valores = {"a": 1.0, "b": 2.0, "c": 100.0}
    trials = {}
    for nombre, valor in valores.items():
        t = study.ask()
        t.report(valor, escalon)
        trials[nombre] = t

    assert trials["a"].should_prune() is False
    assert trials["c"].should_prune() is True
    assert "escalon=" in (podador.motivo_de(trials["c"].number) or "")


def test_motivo_de_es_por_trial_no_un_estado_compartido():
    # Prueba de regresion del bug corregido: antes, `ultimo_motivo` era un
    # unico atributo de instancia que un trial B podia pisar antes de que
    # el trial A lo leyera.
    decisor = _decisor(min_ventanas=4, n_ventanas_totales=12, factor_reduccion=3)
    study, podador = crear_estudio_optuna(
        muestreador=optuna.samplers.RandomSampler(seed=0), decisor_asha=decisor
    )
    escalon = decisor.escalones()[0]
    t_mala = study.ask()
    t_mala.report(100.0, escalon)
    t_buena = study.ask()
    t_buena.report(1.0, escalon)

    assert t_mala.should_prune() is True
    # simula que otro trial "pisa" el pruner despues, antes de leer el motivo
    t_otra_mala = study.ask()
    t_otra_mala.report(200.0, escalon)
    assert t_otra_mala.should_prune() is True

    motivo_mala = podador.motivo_de(t_mala.number)
    assert motivo_mala is not None  # no lo piso el trial que podo despues

    # motivo_de hace pop: una segunda lectura del mismo trial ya no lo tiene
    assert podador.motivo_de(t_mala.number) is None
    # y nunca contamino el motivo de un trial que no debia tenerlo
    assert podador.motivo_de(t_buena.number) is None


def test_jsonl_round_trip(tmp_path: Path):
    podador = PodadorASHAOptuna(_decisor())
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=podador,
    )
    t1 = study.ask()
    t1.suggest_int("p", 0, 5)
    t1.set_user_attr("trial_id", "t1")
    t1.set_user_attr("n_ventanas", 10)
    study.tell(t1, 0.5, state=optuna.trial.TrialState.COMPLETE)

    ruta = tmp_path / "estudio.jsonl"
    volcar_jsonl(study, ruta)

    lineas = ruta.read_text(encoding="utf-8").strip().splitlines()
    assert len(lineas) == 1
    assert lineas[0]

    reanudado = reanudar_estudio(
        ruta,
        espacio=_espacio(),
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=PodadorASHAOptuna(_decisor()),
    )
    (info,) = trials_desde_study(reanudado)
    assert info.atributos["trial_id"] == "t1"
    assert info.valor == 0.5


def test_reanudar_sin_archivo_devuelve_estudio_vacio(tmp_path: Path):
    reanudado = reanudar_estudio(
        tmp_path / "no_existe.jsonl",
        espacio=_espacio(),
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=PodadorASHAOptuna(_decisor()),
    )
    assert reanudado.get_trials(deepcopy=False) == []
