"""0.3-B3 - Pruebas de la compuerta de restricciones fisicas (A1, A2, A3)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pred_engine.aumentacion.errores import PhysicalConstraintError
from pred_engine.aumentacion.restricciones import (
    LimitesLeadTime,
    acotar_lead_time,
    aplicar_restricciones_fisicas,
    limites_lead_time_desde_semilla,
    rectificar_demanda_no_negativa,
    truncar_a_unidades_enteras,
)


# --- A1: clipping asimetrico ------------------------------------------------
def test_clipping_elimina_demanda_negativa_sin_tocar_positivos() -> None:
    entrada = np.array([-3.0, 0.0, 2.5, -0.1, 10.0])
    resultado = rectificar_demanda_no_negativa(entrada)
    assert list(resultado.valores) == [0.0, 0.0, 2.5, 0.0, 10.0]
    assert resultado.n_rectificadas == 2
    assert resultado.porcentaje == pytest.approx(40.0)


def test_clipping_no_muta_la_entrada() -> None:
    entrada = np.array([-1.0, 5.0])
    original = entrada.copy()
    rectificar_demanda_no_negativa(entrada)
    assert np.array_equal(entrada, original)


def test_clipping_rechaza_arreglo_no_1d() -> None:
    with pytest.raises(PhysicalConstraintError):
        rectificar_demanda_no_negativa(np.zeros((2, 2)))


# --- A2: truncamiento entero ----------------------------------------------
def test_truncamiento_no_redondea_hacia_arriba() -> None:
    entrada = np.array([2.99, 0.5, 7.0, 3.2])
    salida = truncar_a_unidades_enteras(entrada)
    assert list(salida) == [2, 0, 7, 3]
    assert salida.dtype == np.int64


def test_truncamiento_usa_valor_absoluto() -> None:
    salida = truncar_a_unidades_enteras(np.array([-2.7, -0.9]))
    assert list(salida) == [2, 0]


# --- A3: limites de lead time -------------------------------------------
def test_limites_se_derivan_de_la_semilla() -> None:
    limites = limites_lead_time_desde_semilla(np.array([2.0, 5.0, 9.5]))
    assert limites == LimitesLeadTime(minimo=2, maximo=10)


def test_limite_inferior_nunca_baja_de_uno() -> None:
    limites = limites_lead_time_desde_semilla(np.array([0.0, 0.4, 3.0]))
    assert limites.minimo == 1


def test_acotar_lead_time_recorta_fuera_de_rango() -> None:
    limites = LimitesLeadTime(minimo=2, maximo=8)
    resultado = acotar_lead_time(np.array([-1.0, 0.0, 5.0, 99.0]), limites)
    assert list(resultado.valores) == [2, 2, 5, 8]
    assert resultado.n_acotadas == 3


def test_limites_invalidos_son_rechazados() -> None:
    with pytest.raises(PhysicalConstraintError):
        LimitesLeadTime(minimo=0, maximo=5)
    with pytest.raises(PhysicalConstraintError):
        LimitesLeadTime(minimo=5, maximo=3)


def test_semilla_vacia_de_lead_time_es_rechazada() -> None:
    with pytest.raises(PhysicalConstraintError):
        limites_lead_time_desde_semilla(np.array([]))


# --- compuerta combinada ------------------------------------------------
def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": ["A", "A", "B"],
            "timestamp": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 1),
            ],
            "demand_qty": [-2.5, 3.9, 12.0],
            "lead_time_days": [0.0, 4.0, 40.0],
        }
    )


def test_compuerta_aplica_las_tres_leyes() -> None:
    limites = LimitesLeadTime(minimo=1, maximo=10)
    resultado = aplicar_restricciones_fisicas(_panel(), limites_lead_time=limites)
    panel = resultado.panel
    assert list(panel["demand_qty"]) == [0, 3, 12]
    assert panel["demand_qty"].dtype == np.int64
    assert list(panel["lead_time_days"]) == [1, 4, 10]
    assert resultado.n_demanda_rectificada == 1
    assert resultado.n_lead_time_acotado == 2


def test_compuerta_no_muta_el_panel_de_entrada() -> None:
    panel = _panel()
    copia = panel.copy(deep=True)
    aplicar_restricciones_fisicas(
        panel, limites_lead_time=LimitesLeadTime(minimo=1, maximo=10)
    )
    pd.testing.assert_frame_equal(panel, copia)


def test_compuerta_rechaza_panel_incompleto() -> None:
    with pytest.raises(PhysicalConstraintError):
        aplicar_restricciones_fisicas(
            pd.DataFrame({"sku_id": ["A"]}),
            limites_lead_time=LimitesLeadTime(minimo=1, maximo=2),
        )


def test_compuerta_rechaza_panel_vacio() -> None:
    vacio = _panel().iloc[0:0]
    with pytest.raises(PhysicalConstraintError):
        aplicar_restricciones_fisicas(
            vacio, limites_lead_time=LimitesLeadTime(minimo=1, maximo=2)
        )
