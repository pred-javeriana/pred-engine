"""Componente 5: la regla mas importante es 'nunca podar antes de min_ventanas'."""

from __future__ import annotations

from pred_engine.comun.dataclasses.validacion_temporal import (
    EstadoParcial,
    ResultadoVentana,
    VentanaTemporal,
)
from pred_engine.optimizacion.optimizadores.HPO.asha import (
    AsignadorRecursosASHA,
    Decision,
)
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda


def _estado(n_evaluadas: int, valor_parcial: float) -> EstadoParcial:
    ventana = VentanaTemporal(
        indice=n_evaluadas - 1,
        inicio_train=0,
        fin_train=10,
        inicio_val=10,
        fin_val=11,
    )
    resultado = ResultadoVentana(
        ventana=ventana,
        metricas={"mase": valor_parcial},
        y_train=None,  # type: ignore[arg-type]
        y_real=None,  # type: ignore[arg-type]
        y_pred=None,  # type: ignore[arg-type]
        duracion_s=0.0,
        fallo=None,
    )
    return EstadoParcial(
        ultima=resultado,
        n_evaluadas=n_evaluadas,
        n_totales=30,
        valor_parcial=valor_parcial,
    )


def test_ningun_trial_se_poda_antes_de_min_ventanas():
    reglas = ReglasPoda(min_ventanas=4)
    asignador = AsignadorRecursosASHA(reglas, n_ventanas_totales=30)
    for n in range(1, 4):
        asignador.registrar("t1", _estado(n, valor_parcial=1000.0))
        decision, _ = asignador.decidir("t1")
        assert decision is Decision.CONTINUAR


def test_n_ventanas_totales_menor_a_min_ventanas_es_invalido():
    import pytest

    with pytest.raises(ValueError):
        AsignadorRecursosASHA(ReglasPoda(min_ventanas=4), n_ventanas_totales=2)


def test_escalones_arrancan_en_min_ventanas_y_terminan_en_el_total():
    reglas = ReglasPoda(min_ventanas=4, factor_reduccion=3)
    asignador = AsignadorRecursosASHA(reglas, n_ventanas_totales=30)
    assert asignador.escalones()[0] == 4
    assert asignador.escalones()[-1] == 30
    assert list(asignador.escalones()) == sorted(asignador.escalones())


def test_promocion_del_mejor_tercio_en_un_escalon():
    reglas = ReglasPoda(min_ventanas=4, factor_reduccion=3)
    asignador = AsignadorRecursosASHA(reglas, n_ventanas_totales=12)
    escalon = asignador.escalones()[0]
    valores = {f"t{i}": float(i) for i in range(6)}
    for trial_id, valor in valores.items():
        asignador.registrar(trial_id, _estado(escalon, valor))

    assert asignador.decidir("t0")[0] is Decision.CONTINUAR
    assert asignador.decidir("t5")[0] is Decision.PODAR


def test_motivo_de_poda_incluye_escalon_y_valores():
    reglas = ReglasPoda(min_ventanas=4, factor_reduccion=3)
    asignador = AsignadorRecursosASHA(reglas, n_ventanas_totales=12)
    escalon = asignador.escalones()[0]
    for trial_id, valor in {"a": 1.0, "b": 2.0, "c": 100.0}.items():
        asignador.registrar(trial_id, _estado(escalon, valor))
    decision, motivo = asignador.decidir("c")
    assert decision is Decision.PODAR
    assert motivo is not None
    assert f"escalon={escalon}" in motivo


def test_asincronia_orden_de_llegada_no_cambia_el_veredicto():
    reglas = ReglasPoda(min_ventanas=4, factor_reduccion=3)
    a = AsignadorRecursosASHA(reglas, n_ventanas_totales=12)
    b = AsignadorRecursosASHA(reglas, n_ventanas_totales=12)
    escalon = a.escalones()[0]
    valores = {"t0": 1.0, "t1": 5.0, "t2": 9.0}

    for trial_id in ("t0", "t1", "t2"):
        a.registrar(trial_id, _estado(escalon, valores[trial_id]))
    for trial_id in ("t2", "t0", "t1"):
        b.registrar(trial_id, _estado(escalon, valores[trial_id]))

    for trial_id in valores:
        assert a.decidir(trial_id)[0] == b.decidir(trial_id)[0]


def test_sin_suficientes_pares_no_decide_todavia():
    reglas = ReglasPoda(min_ventanas=4, factor_reduccion=3)
    asignador = AsignadorRecursosASHA(reglas, n_ventanas_totales=12)
    escalon = asignador.escalones()[0]
    asignador.registrar("solo", _estado(escalon, 1000.0))
    assert asignador.decidir("solo")[0] is Decision.CONTINUAR
