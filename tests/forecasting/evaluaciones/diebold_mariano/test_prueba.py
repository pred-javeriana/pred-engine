"""Diebold-Mariano HLN, intervalo de r y correccion BH (ADR-03-011, ADR-023)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import acovf
from tests._evaluaciones import metricas_candidato, par_fieller_disjunto

from pred_engine.forecasting.evaluaciones.calculo_errores import FAMILIA_LINEA_BASE
from pred_engine.forecasting.evaluaciones.diebold_mariano import (
    PruebaDM,
    corregir_multiplicidad,
    intervalo_razon,
    probar_contra_linea_base,
    probar_pares,
    prueba_hln,
)

_T = np.arange(40)
_LINEA_BASE = list(np.sqrt(1.0 + 0.4 * np.cos(0.7 * _T)))


def _par(media: float):
    """Campeon con perdida media `media` frente a una linea base de perdida 1."""
    rmses = list(np.sqrt(media + 0.3 * np.sin(1.1 * _T)))
    return (
        metricas_candidato("ml", 0.8, rmses=rmses),
        metricas_candidato(FAMILIA_LINEA_BASE, 1.0, rmses=_LINEA_BASE),
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


def test_minimo_de_origenes_comunes_predeclarado_no_emite_p_valor_antes() -> None:
    campeon = metricas_candidato("ml", 0.8, n=6, rmses=[0.5, 1.0, 0.2, 0.9, 0.4, 3.0])
    base = metricas_candidato(FAMILIA_LINEA_BASE, 1.0, n=4, rmses=[1.0] * 4)
    corta = probar_contra_linea_base(campeon, base, n_min_ventanas=5)
    suficiente = probar_contra_linea_base(campeon, base, n_min_ventanas=4)
    assert (corta.n_ventanas, corta.horizonte, corta.causa) == (4, 1, "pocas_ventanas")
    assert corta.p_valor is None
    assert suficiente.estadistico == pytest.approx(-2.0936083029259844)
    assert suficiente.p_valor == pytest.approx(0.12733814363872983)


@pytest.mark.parametrize("minimo", [1, 0, True, 2.5])
def test_minimo_de_origenes_invalido_no_se_acepta(minimo) -> None:
    with pytest.raises(ValueError, match="n_min_ventanas"):
        probar_contra_linea_base(
            metricas_candidato("ml", 0.8),
            metricas_candidato(FAMILIA_LINEA_BASE, 1.0),
            n_min_ventanas=minimo,
        )


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
    assert {corregidas[s].alfa for s in "ABC"} == {0.035}
    assert corregidas["D"] == pruebas["D"]


def test_bh_by_ajusta_con_benjamini_yekutieli() -> None:
    pruebas = {
        s: PruebaDM(40, 1, estadistico=-2.0, p_valor=p)
        for s, p in zip("ABC", (0.01, 0.02, 0.03), strict=True)
    }
    corregidas = corregir_multiplicidad(pruebas, alfa=0.05, metodo="fdr_by")
    rechazos, ajustados, _, _ = multipletests(
        [0.01, 0.02, 0.03], alpha=0.05, method="fdr_by"
    )
    assert [corregidas[s].p_ajustado for s in "ABC"] == pytest.approx(list(ajustados))
    assert [corregidas[s].significativa for s in "ABC"] == list(rechazos)


@pytest.mark.parametrize("media", [0.6, 0.95])
def test_intervalo_de_r_invierte_hln(media: float) -> None:
    campeon, base = _par(media)
    intervalo = intervalo_razon(campeon, base, nivel=0.95)
    assert intervalo is not None and intervalo.superior is not None
    a = np.square([v.metricas.rmse for v in campeon.por_ventana])
    b = np.square(_LINEA_BASE)
    # En r la media del diferencial a - r^2 b es 0: el intervalo contiene a r.
    r = math.sqrt(a.mean() / b.mean())
    assert intervalo.inferior < r < intervalo.superior
    # En cada extremo el estadistico HLN vale el valor critico.
    critico = stats.t.ppf(0.975, df=39)
    for rho in (intervalo.inferior, intervalo.superior):
        prueba = prueba_hln(a - rho**2 * b, 1)
        assert abs(prueba.estadistico) == pytest.approx(critico)
    # Excluye 1 justo cuando la prueba rechaza al mismo nivel.
    p_valor = probar_pares(campeon, base).p_valor
    assert p_valor is not None
    assert (intervalo.superior < 1) is (p_valor < 0.05)


def test_intervalo_sin_cota_superior_si_sn_no_se_distingue_de_cero() -> None:
    campeon = metricas_candidato(
        "ml", 0.8, n=8, rmses=[0.5, 0.4, 0.6, 0.5, 0.45, 0.55, 0.5, 0.5]
    )
    linea_base = metricas_candidato(
        FAMILIA_LINEA_BASE, 1.0, n=8, rmses=[3.0] + [0.01] * 7
    )
    intervalo = intervalo_razon(campeon, linea_base, nivel=0.95)
    assert intervalo is not None
    assert intervalo.superior is None and intervalo.inferior > 0


def test_intervalo_con_dos_tramos_los_conserva_y_excluye_1() -> None:
    campeon, base = par_fieller_disjunto()
    a = np.square([v.metricas.rmse for v in campeon.por_ventana])
    b = np.square([v.metricas.rmse for v in base.por_ventana])
    p_valor = probar_pares(campeon, base, n_min_ventanas=30).p_valor
    assert p_valor is not None and p_valor < 1e-80
    intervalo = intervalo_razon(campeon, base, nivel=0.95)
    assert intervalo is not None
    # [0, 0.9923] U [1.0257, inf): la cuadratica abre hacia abajo con dos raices
    # positivas, y la prueba rechaza los valores entre ellas.
    assert intervalo.inferior == 0.0
    assert intervalo.superior == pytest.approx(0.992283906333525)
    assert intervalo.inferior_cola == pytest.approx(1.0256630707673464)
    critico = stats.t.ppf(0.975, df=39)
    for rho in (intervalo.superior, intervalo.inferior_cola):
        prueba = prueba_hln(a - rho**2 * b, 7)
        assert prueba.estadistico is not None
        assert abs(prueba.estadistico) == pytest.approx(critico)
    r = math.sqrt(a.mean() / b.mean())
    for rho in (0.5, r, 1.0, 2.0):
        prueba = prueba_hln(a - rho**2 * b, 7)
        assert prueba.p_valor is not None
        assert intervalo.contiene(rho) is (prueba.p_valor >= 0.05)
    assert intervalo.contiene(r) and not intervalo.contiene(1.0)


def test_intervalo_a_mayor_nivel_es_mas_ancho_y_sin_prueba_no_existe() -> None:
    campeon, base = _par(0.6)
    al_95 = intervalo_razon(campeon, base, nivel=0.95)
    al_99 = intervalo_razon(campeon, base, nivel=0.99)
    assert al_95 is not None and al_99 is not None
    assert al_95.superior is not None and al_99.superior is not None
    assert al_99.inferior < al_95.inferior and al_95.superior < al_99.superior
    corto = metricas_candidato("ml", 0.8, n=1)
    assert intervalo_razon(corto, base, nivel=0.95) is None
    with pytest.raises(ValueError, match="nivel"):
        intervalo_razon(campeon, base, nivel=1.0)
