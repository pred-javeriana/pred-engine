"""Componente 5: la regla mas importante es 'nunca podar antes de min_ventanas'.

`DecisorASHA` es el algoritmo puro (sin Optuna): recibe numeros y devuelve
una `DecisionPoda`. Estas pruebas lo ejercitan directamente, sin construir
ningun `optuna.Study` -- eso es justamente lo que prueba que el algoritmo
esta desacoplado del backend. La integracion con Optuna (`PodadorASHAOptuna`,
`should_prune()` real) se prueba en `test_adaptador_optuna.py`.
"""

from __future__ import annotations

import pytest

from pred_engine.optimizacion.optimizadores.HPO.asha import DecisionPoda, DecisorASHA
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda


def test_ningun_trial_se_poda_antes_de_min_ventanas():
    decisor = DecisorASHA(ReglasPoda(min_ventanas=4), n_ventanas_totales=30)
    for n in range(1, 4):
        decision = decisor.decidir(
            n_evaluadas=n, numero_trial=0, valores_competidores_en_escalon={0: 1000.0}
        )
        assert decision == DecisionPoda(podar=False)


def test_n_ventanas_totales_menor_a_min_ventanas_es_invalido():
    with pytest.raises(ValueError):
        DecisorASHA(ReglasPoda(min_ventanas=4), n_ventanas_totales=2)


def test_escalones_arrancan_en_min_ventanas_y_terminan_en_el_total():
    decisor = DecisorASHA(
        ReglasPoda(min_ventanas=4, factor_reduccion=3), n_ventanas_totales=30
    )
    assert decisor.escalones()[0] == 4
    assert decisor.escalones()[-1] == 30
    assert list(decisor.escalones()) == sorted(decisor.escalones())


def test_promocion_del_mejor_tercio_en_un_escalon():
    decisor = DecisorASHA(
        ReglasPoda(min_ventanas=4, factor_reduccion=3), n_ventanas_totales=12
    )
    escalon = decisor.escalones()[0]
    competidores = {i: float(i) for i in range(6)}  # 0.0 .. 5.0

    decision_mejor = decisor.decidir(
        n_evaluadas=escalon,
        numero_trial=0,
        valores_competidores_en_escalon=competidores,
    )
    decision_peor = decisor.decidir(
        n_evaluadas=escalon,
        numero_trial=5,
        valores_competidores_en_escalon=competidores,
    )
    assert decision_mejor.podar is False
    assert decision_peor.podar is True


def test_motivo_de_poda_incluye_escalon_y_valores():
    decisor = DecisorASHA(
        ReglasPoda(min_ventanas=4, factor_reduccion=3), n_ventanas_totales=12
    )
    escalon = decisor.escalones()[0]
    competidores = {0: 1.0, 1: 2.0, 2: 100.0}

    decision = decisor.decidir(
        n_evaluadas=escalon,
        numero_trial=2,
        valores_competidores_en_escalon=competidores,
    )
    assert decision.podar is True
    assert decision.motivo is not None
    assert f"escalon={escalon}" in decision.motivo


def test_asincronia_orden_de_llegada_no_cambia_el_veredicto():
    reglas = ReglasPoda(min_ventanas=4, factor_reduccion=3)
    valores = {0: 1.0, 1: 5.0, 2: 9.0}

    decisor_a = DecisorASHA(reglas, n_ventanas_totales=12)
    escalon = decisor_a.escalones()[0]
    veredicto_a = {
        numero: decisor_a.decidir(
            n_evaluadas=escalon,
            numero_trial=numero,
            valores_competidores_en_escalon=valores,
        ).podar
        for numero in (0, 1, 2)
    }

    decisor_b = DecisorASHA(reglas, n_ventanas_totales=12)
    veredicto_b = {
        numero: decisor_b.decidir(
            n_evaluadas=escalon,
            numero_trial=numero,
            valores_competidores_en_escalon=valores,
        ).podar
        for numero in (2, 0, 1)  # mismo conjunto, orden de evaluacion distinto
    }

    assert veredicto_a == veredicto_b


def test_sin_suficientes_competidores_no_decide_todavia():
    decisor = DecisorASHA(
        ReglasPoda(min_ventanas=4, factor_reduccion=3), n_ventanas_totales=12
    )
    escalon = decisor.escalones()[0]
    decision = decisor.decidir(
        n_evaluadas=escalon, numero_trial=0, valores_competidores_en_escalon={0: 1000.0}
    )
    assert decision.podar is False


def test_trial_ausente_del_escalon_no_se_poda():
    decisor = DecisorASHA(
        ReglasPoda(min_ventanas=4, factor_reduccion=3), n_ventanas_totales=12
    )
    escalon = decisor.escalones()[0]
    decision = decisor.decidir(
        n_evaluadas=escalon,
        numero_trial=99,  # no esta en el dict de competidores
        valores_competidores_en_escalon={0: 1.0, 1: 2.0},
    )
    assert decision.podar is False
