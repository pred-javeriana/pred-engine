"""Regla de seguridad #3: poda semantica (`es_degenerada`), version simplificada.

`prediccion_constante` como regla generica fue DESCARTADA deliberadamente:
predecir el mismo valor en todos los pasos del horizonte es el
comportamiento normal de baselines legitimos (naive, media). Un umbral
generico produjo un falso positivo catastrofico en pruebas de integracion
(ver `poda.py`). Solo quedan las tres condiciones inequivocamente invalidas.
"""

from __future__ import annotations

import numpy as np

from pred_engine.optimizacion.optimizadores.HPO.poda import es_degenerada


def test_prediccion_constante_no_activa_la_poda_por_si_sola():
    y_pred = np.full(5, 3.0)
    y_train = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    es_mala, motivo = es_degenerada(y_pred, y_train=y_train)
    assert es_mala is False
    assert motivo is None


def test_prediccion_nula_con_demanda_observada():
    y_pred = np.zeros(5)
    y_train = np.array([0.0, 0.0, 1.0, 0.0, 2.0])
    es_mala, motivo = es_degenerada(y_pred, y_train=y_train)
    assert es_mala is True
    assert motivo == "prediccion_nula"


def test_prediccion_nula_sin_demanda_observada_no_es_degenerada():
    y_pred = np.zeros(5)
    y_train = np.zeros(10)
    es_mala, _ = es_degenerada(y_pred, y_train=y_train)
    assert es_mala is False


def test_prediccion_no_finita():
    y_pred = np.array([1.0, np.nan, 2.0])
    es_mala, motivo = es_degenerada(y_pred, y_train=np.array([1.0, 2.0]))
    assert es_mala is True
    assert motivo == "prediccion_no_finita"


def test_demanda_negativa_es_degenerada():
    y_pred = np.array([1.0, -0.5, 2.0])
    es_mala, motivo = es_degenerada(y_pred, y_train=np.array([1.0, 2.0, 3.0]))
    assert es_mala is True
    assert motivo == "demanda_negativa"


def test_prediccion_variable_no_es_degenerada():
    y_pred = np.array([1.0, 2.0, 1.5])
    y_train = np.array([1.0, 3.0, 2.0])
    es_mala, motivo = es_degenerada(y_pred, y_train=y_train)
    assert es_mala is False
    assert motivo is None
