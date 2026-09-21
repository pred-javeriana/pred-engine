"""Espacio de busqueda de ML: rangos, restricciones y huella estable."""

from __future__ import annotations

import numpy as np

from pred_engine.optimizacion.optimizadores.HPO.espacio import Flotante
from pred_engine.optimizacion.optimizadores.modelos_machine_learning import (
    EspacioML,
    construir_espacio_ml,
    min_train_recomendado_ml,
)

_ESPERADOS = {
    "lags",
    "n_estimators",
    "max_depth",
    "learning_rate",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
}


def test_el_espacio_declara_los_hiperparametros_de_la_spec_2_5():
    nombres = {p.nombre for p in construir_espacio_ml().parametros}
    assert nombres == _ESPERADOS


def test_learning_rate_y_regularizacion_van_en_escala_log():
    logs = {
        p.nombre
        for p in construir_espacio_ml().parametros
        if isinstance(p, Flotante) and p.log
    }
    assert {"learning_rate", "reg_alpha", "reg_lambda", "min_child_weight"} <= logs


def test_toda_configuracion_muestreada_respeta_rangos_y_restricciones():
    cfg = EspacioML(costo_max=400)
    espacio = construir_espacio_ml(cfg)
    rng = np.random.default_rng(0)
    for _ in range(200):
        c = espacio.muestrear(rng)
        assert cfg.lags_min <= c["lags"] <= cfg.lags_max
        assert cfg.n_estimators_min <= c["n_estimators"] <= cfg.n_estimators_max
        assert cfg.max_depth_min <= c["max_depth"] <= cfg.max_depth_max
        assert cfg.learning_rate_min <= c["learning_rate"] <= cfg.learning_rate_max
        assert cfg.subsample_min <= c["subsample"] <= 1.0
        assert c["n_estimators"] * c["max_depth"] <= cfg.costo_max
        assert espacio.es_valida(c)


def test_descripcion_canonica_es_identica_entre_construcciones():
    a = construir_espacio_ml().descripcion_canonica()
    b = construir_espacio_ml().descripcion_canonica()
    assert a == b


def test_descripcion_canonica_cambia_si_cambia_la_configuracion():
    base = construir_espacio_ml().descripcion_canonica()
    otro = construir_espacio_ml(EspacioML(max_depth_max=5)).descripcion_canonica()
    assert base != otro


def test_min_train_recomendado_cubre_el_mayor_lag():
    cfg = EspacioML(lags_max=14, m=7)
    assert min_train_recomendado_ml(cfg) >= cfg.lags_max + 10
    assert min_train_recomendado_ml(EspacioML(m=1)) > 0
