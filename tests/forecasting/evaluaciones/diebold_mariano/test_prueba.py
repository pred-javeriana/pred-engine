"""Diebold-Mariano HLN y correccion BH (ADR-03-008, alternativa 2)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats
from statsmodels.tsa.stattools import acovf
from tests._evaluaciones import metricas_candidato

from pred_engine.forecasting.evaluaciones.calculo_errores import FAMILIA_LINEA_BASE
from pred_engine.forecasting.evaluaciones.diebold_mariano import (
    PruebaDM,
    corregir_multiplicidad,
    probar_contra_linea_base,
    prueba_hln,
)


def test_hln_coincide_con_el_calculo_a_mano() -> None:
    # media 0.5, gamma0 = 1.25, DM = 0.5 / sqrt(1.25/4), factor HLN sqrt(3/4).
    prueba = prueba_hln(np.array([1.0, -1.0, 2.0, 0.0]), 1)
    assert prueba.estadistico == pytest.approx(math.sqrt(0.6))
    assert prueba.p_valor == pytest.approx(2 * stats.t.sf(math.sqrt(0.6), df=3))
    assert (prueba.n_ventanas, prueba.horizonte, prueba.causa) == (4, 1, None)


def test_con_horizonte_h_suma_las_autocovarianzas_hasta_h_menos_1() -> None:
    d = np.array([-0.2, -0.5, -0.9, -0.4, 0.1, -0.3, -0.7, -1.0, -0.6, -0.2])
    t, h = len(d), 2
    gamma = acovf(d, demean=True, fft=False)
    dm = d.mean() / math.sqrt((gamma[0] + 2 * gamma[1]) / t)
    # (T + 1 - 2h + h(h-1)/T) / T == (T - h)(T - h + 1) / T^2
    esperado = dm * math.sqrt((t - h) * (t - h + 1)) / t
    prueba = prueba_hln(d, h)
    assert prueba.estadistico == pytest.approx(esperado)
    assert prueba.p_valor == pytest.approx(2 * stats.t.sf(abs(esperado), df=t - 1))


def test_campeon_con_menor_perdida_da_estadistico_negativo() -> None:
    prueba = prueba_hln(np.array([-1.0, -0.5, -1.5, -0.8, -1.2]), 1)
    assert prueba.estadistico is not None and prueba.estadistico < 0


@pytest.mark.parametrize(
    ("diferencias", "horizonte", "causa"),
    [
        ([1.0], 1, "pocas_ventanas"),
        ([1.0, 2.0, 3.0], 3, "pocas_ventanas"),
        ([0.4, 0.4, 0.4, 0.4], 1, "varianza_no_positiva"),
        # Autocovarianza de rezago 1 tan negativa que la suma no es positiva.
        ([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], 2, "varianza_no_positiva"),
    ],
)
def test_sin_prueba_calculable_queda_la_causa(
    diferencias: list[float], horizonte: int, causa: str
) -> None:
    prueba = prueba_hln(np.array(diferencias), horizonte)
    assert prueba.causa == causa
    assert prueba.estadistico is None and prueba.p_valor is None


def test_horizonte_menor_que_uno_se_rechaza() -> None:
    with pytest.raises(ValueError):
        prueba_hln(np.array([1.0, 2.0, 3.0]), 0)


def test_contra_la_linea_base_usa_perdida_cuadratica_en_ventanas_comunes() -> None:
    campeon = metricas_candidato("ml", 0.8, n=6, rmses=[0.5, 1.0, 0.2, 0.9, 0.4, 3.0])
    # La linea base no tiene la sexta ventana: queda fuera de la prueba.
    base = metricas_candidato(FAMILIA_LINEA_BASE, 1.0, n=5, rmses=[1.0] * 5)
    prueba = probar_contra_linea_base(campeon, base)
    esperado = prueba_hln(np.array([0.25, 1.0, 0.04, 0.81, 0.16]) - 1.0, 1)
    assert (prueba.n_ventanas, prueba.horizonte) == (5, 1)
    assert prueba.estadistico == pytest.approx(esperado.estadistico)
    assert prueba.p_valor == pytest.approx(esperado.p_valor)


def test_sin_ventanas_comunes_no_se_prueba() -> None:
    campeon = metricas_candidato("ml", None, n=0)
    base = metricas_candidato(FAMILIA_LINEA_BASE, 1.0)
    assert probar_contra_linea_base(campeon, base).causa == "sin_ventanas_comunes"


def test_bh_ajusta_solo_las_pruebas_calculadas() -> None:
    pruebas = {
        "A": PruebaDM(40, 1, estadistico=-3.0, p_valor=0.01),
        "B": PruebaDM(40, 1, estadistico=-2.0, p_valor=0.04),
        "C": PruebaDM(40, 1, estadistico=2.2, p_valor=0.03),
        "D": PruebaDM(0, 0, causa="datos_sinteticos"),
    }
    corregidas = corregir_multiplicidad(pruebas, alfa=0.035)
    # BH con m=3: 0.01*3/1, min(0.03*3/2, 0.04), 0.04*3/3.
    assert {s: corregidas[s].p_ajustado for s in "ABC"} == pytest.approx(
        {"A": 0.03, "B": 0.04, "C": 0.04}
    )
    assert [corregidas[s].significativa for s in "ABC"] == [True, False, False]
    assert corregidas["D"] == pruebas["D"]
