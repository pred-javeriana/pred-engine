"""Componente 9 (parcial): 'podado' y 'fallido' deben ser distinguibles, con motivo.

`instantanea_desde_estudio` traduce `InfoTrial` (el snapshot ya resuelto por
el adaptador de backend, ver `adaptador_optuna.py`) al contrato externo
estable (`Trial`/`ResultadoEstudio`). Estas pruebas construyen `InfoTrial`
a mano -- no necesitan un `optuna.Study` real, que es justamente lo que
demuestra que esta funcion esta desacoplada del backend. Las pruebas contra
un `Study` real (incluida la persistencia JSONL) viven en
`test_adaptador_optuna.py`.
"""

from __future__ import annotations

from pred_engine.comun.dataclasses.validacion_temporal import VentanaTemporal
from pred_engine.optimizacion.optimizadores.HPO.contratos import InfoTrial
from pred_engine.optimizacion.optimizadores.HPO.registro import (
    instantanea_desde_estudio,
)


def _ventanas() -> tuple[VentanaTemporal, ...]:
    return (
        VentanaTemporal(
            indice=0, inicio_train=0, fin_train=10, inicio_val=10, fin_val=11
        ),
    )


def test_podado_y_fallido_son_estados_distintos_con_motivo_no_nulo():
    trials = [
        InfoTrial(
            numero=0,
            estado="podado",
            valor=2.0,
            parametros={"p": 1},
            atributos={
                "trial_id": "t1",
                "n_ventanas": 4,
                "motivo": "ventana=4 valor=2.0 asha",
            },
        ),
        InfoTrial(
            numero=1,
            estado="fallido",
            valor=None,
            parametros={"p": 2},
            atributos={"trial_id": "t2", "motivo": "excepcion_de_ajuste"},
        ),
    ]

    resultado = instantanea_desde_estudio(
        trials, ventanas=_ventanas(), metrica_objetivo="mase", seed=0
    )
    trials_por_id = {t.id: t for t in resultado.trials}

    assert trials_por_id["t1"].estado == "podado"
    assert trials_por_id["t1"].valor == 2.0  # el valor del podado se preserva
    assert trials_por_id["t1"].motivo is not None
    assert "asha" in trials_por_id["t1"].motivo
    assert trials_por_id["t2"].estado == "fallido"
    assert trials_por_id["t2"].valor is None
    assert trials_por_id["t2"].motivo == "excepcion_de_ajuste"
    assert trials_por_id["t1"].estado != trials_por_id["t2"].estado


def test_instantanea_elige_el_menor_valor_entre_completados():
    trials = [
        InfoTrial(
            numero=0, estado="completado", valor=2.0,
            parametros={"p": 1}, atributos={"trial_id": "t1"},
        ),
        InfoTrial(
            numero=1, estado="completado", valor=0.5,
            parametros={"p": 2}, atributos={"trial_id": "t2"},
        ),
    ]
    resultado = instantanea_desde_estudio(
        trials, ventanas=_ventanas(), metrica_objetivo="mase", seed=0
    )
    assert resultado.mejor is not None
    assert resultado.mejor.id == "t2"
    assert resultado.n_completados == 2


def test_instantanea_mejor_es_none_sin_completados():
    trials = [
        InfoTrial(
            numero=0, estado="fallido", valor=None,
            parametros={"p": 1}, atributos={"trial_id": "t1", "motivo": "x"},
        ),
    ]
    resultado = instantanea_desde_estudio(
        trials, ventanas=_ventanas(), metrica_objetivo="mase", seed=0
    )
    assert resultado.mejor is None
    assert resultado.n_fallidos == 1
    assert resultado.n_completados == 0
    assert resultado.n_podados == 0


def test_instantanea_no_cuenta_como_completado_un_trial_sin_valor():
    # Defensivo: un "completado" sin valor (no deberia darse en la practica,
    # pero si el backend lo produce, no debe entrar a la carrera por el
    # minimo NI al conteo de completados -- ambos exigen `valor is not None`).
    trials = [
        InfoTrial(
            numero=0, estado="completado", valor=None,
            parametros={}, atributos={"trial_id": "t1"},
        ),
    ]
    resultado = instantanea_desde_estudio(
        trials, ventanas=_ventanas(), metrica_objetivo="mase", seed=0
    )
    assert resultado.mejor is None
    assert resultado.n_completados == 0


def test_instantanea_usa_numero_de_trial_si_no_hay_trial_id():
    trials = [
        InfoTrial(
            numero=7, estado="completado", valor=1.0, parametros={}, atributos={}
        ),
    ]
    resultado = instantanea_desde_estudio(
        trials, ventanas=_ventanas(), metrica_objetivo="mase", seed=0
    )
    assert resultado.trials[0].id == "7"


def test_instantanea_conserva_ventanas_metrica_y_seed():
    resultado = instantanea_desde_estudio(
        [], ventanas=_ventanas(), metrica_objetivo="mase", seed=42, familia="clasicos"
    )
    assert resultado.ventanas == _ventanas()
    assert resultado.metrica_objetivo == "mase"
    assert resultado.seed == 42
    assert resultado.mejor is None
    assert resultado.trials == ()
