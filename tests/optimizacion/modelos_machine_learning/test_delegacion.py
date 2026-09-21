"""La familia ML es un wrapper delgado: no reimplementa TPE/ASHA ni toca Optuna."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import pred_engine.comun.modelos.modelos_machine_learning as paquete_modelo
import pred_engine.optimizacion.optimizadores.modelos_machine_learning as paquete_sel
import pred_engine.optimizacion.optimizadores.seleccion_hpo as nucleo
import pred_engine.optimizacion.router as paquete_router
from pred_engine.comun.modelos.modelos_machine_learning import LightGBMForecaster
from pred_engine.optimizacion.optimizadores.modelos_machine_learning import (
    EspacioML,
    seleccionar_configuracion_ml,
)

_METODOS_DE_BUCLE_ASK_TELL = {"ask", "tell", "suggest_int", "suggest_float"}


def _archivos(paquete) -> list[Path]:
    return sorted(Path(paquete.__file__).parent.glob("*.py"))


def _importados(ruta: Path) -> set[str]:
    modulos: set[str] = set()
    for nodo in ast.walk(ast.parse(ruta.read_text())):
        if isinstance(nodo, ast.Import):
            modulos.update(a.name for a in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            modulos.add(nodo.module)
    return modulos


@pytest.mark.parametrize("paquete", [paquete_sel, paquete_modelo])
def test_la_familia_ml_no_importa_optuna(paquete):
    for ruta in _archivos(paquete):
        assert not any(m.split(".")[0] == "optuna" for m in _importados(ruta)), ruta


def test_la_familia_ml_no_tiene_bucle_ask_tell_propio():
    for ruta in _archivos(paquete_sel):
        llamadas = {
            n.func.attr
            for n in ast.walk(ast.parse(ruta.read_text()))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert not (llamadas & _METODOS_DE_BUCLE_ASK_TELL), ruta


def test_el_router_no_conoce_ml_ni_hpo():
    prohibidos = {"lightgbm", "optuna"}
    for ruta in _archivos(paquete_router):
        for modulo in _importados(ruta):
            assert modulo.split(".")[0] not in prohibidos, ruta
            assert "optimizadores" not in modulo, ruta


def test_toda_la_optimizacion_pasa_por_ejecutar_estudio(monkeypatch):
    llamadas: list[dict] = []
    real = nucleo.ejecutar_estudio

    def _espia(y, espacio, fabrica, **kw):
        llamadas.append({"espacio": espacio, "fabrica": fabrica, **kw})
        return real(y, espacio, fabrica, **kw)

    monkeypatch.setattr(nucleo, "ejecutar_estudio", _espia)
    rng = np.random.default_rng(0)
    y = 10 + 5 * np.sin(np.arange(60) * 2 * np.pi / 7) + rng.normal(0, 1, 60)
    chico = EspacioML(
        lags_min=3,
        lags_max=5,
        n_estimators_min=5,
        n_estimators_max=10,
        max_depth_min=2,
        max_depth_max=3,
    )
    seleccionar_configuracion_ml(y, sku_id="S1", espacio=chico, n_trials=3)

    assert len(llamadas) == 1
    llamada = llamadas[0]
    assert llamada["familia"] == "ml"
    assert llamada["sku_id"] == "S1"
    assert llamada["n_trials"] == 3
    assert {p.nombre for p in llamada["espacio"].parametros} >= {
        "lags",
        "n_estimators",
        "learning_rate",
    }
    # La fabrica que recibe el motor produce el forecaster de LightGBM, y le
    # inyecta la estacionalidad (`m`) que el espacio no busca.
    modelo = llamada["fabrica"]({"lags": 4}, seed=1)
    assert isinstance(modelo, LightGBMForecaster)
    assert modelo.estacionalidad == chico.m
