"""El DSL comun recibe limites validos de arquitectura, entrenamiento y costo."""

from dataclasses import replace

import numpy as np
import pytest

from pred_engine.comun.modelos.modelos_deep_learning import fabrica_dl
from pred_engine.comun.modelos.modelos_deep_learning.mlp import numero_parametros
from pred_engine.optimizacion.optimizadores.modelos_deep_learning import (
    EspacioDL,
    construir_espacio_dl,
    min_train_recomendado_dl,
)


def test_muestreo_acotado_y_todos_los_candidatos_se_pueden_construir():
    cfg = EspacioDL()
    espacio = construir_espacio_dl(cfg)
    rng = np.random.default_rng(19)
    for _ in range(100):
        c = espacio.muestrear(rng)
        assert set(c) == {
            "lags",
            "capas",
            "unidades",
            "epochs",
            "batch_size",
            "learning_rate",
            "dropout",
            "l2",
        }
        assert espacio.es_valida(c)
        assert (
            numero_parametros(c["lags"], c["capas"], c["unidades"]) * c["epochs"]
            <= cfg.costo_max
        )
        modelo = fabrica_dl(c, seed=0)
        assert modelo.capas >= 2
    assert min_train_recomendado_dl(cfg) == cfg.lags[1] + 5


def test_restriccion_rechaza_esquina_cara_y_su_cota_entra_en_huella():
    cfg = EspacioDL()
    espacio = construir_espacio_dl(cfg)
    cara = {"lags": 14, "capas": 3, "unidades": 32, "epochs": 50}
    assert not espacio.es_valida(cara)
    otro = construir_espacio_dl(replace(cfg, costo_max=200_000))
    assert otro.es_valida(cara)
    assert espacio.descripcion_canonica() != otro.descripcion_canonica()
    assert (
        espacio.descripcion_canonica()
        == construir_espacio_dl(cfg).descripcion_canonica()
    )


@pytest.mark.parametrize(
    "kw",
    [
        {"lags": (3, 2)},
        {"lags": (0, 1)},
        {"lags": (2.5, 3.5)},
        {"capas": (1, 2)},
        {"epochs": (0, 10)},
        {"batch_size": (0, 1)},
        {"learning_rate": (0, 0.1)},
        {"dropout": (0, 1)},
        {"l2": (0, 1)},
        {"unidades": (1, 0)},
        {"costo_max": 0},
        {"costo_max": 1},
        {"learning_rate": (0.1, np.nan)},
    ],
)
def test_espacios_imposibles_fallan_antes_de_hpo(kw):
    with pytest.raises(ValueError):
        EspacioDL(**kw)
