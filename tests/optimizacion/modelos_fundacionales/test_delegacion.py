"""La familia fundacional no se optimiza: sin HPO, sin Walk-Forward, sin torch."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import pred_engine.comun.modelos.modelos_fundacionales as paquete_modelo
import pred_engine.optimizacion.optimizadores.modelos_fundacionales as paquete_sel
import pred_engine.optimizacion.router as paquete_router

_METODOS_DE_BUCLE_ASK_TELL = {"ask", "tell", "suggest_int", "suggest_float"}
# Nada de busqueda ni evaluacion: la evaluacion es del Modulo 3.
_PROHIBIDOS_EN_LA_ESTRATEGIA = (
    "optuna",
    "torch",
    "chronos",
    "pred_engine.optimizacion.optimizadores.HPO",
    "pred_engine.optimizacion.optimizadores.seleccion_hpo",
    "pred_engine.comun.walkforward",
    "pred_engine.comun.modelos.modelos_fundacionales.pipeline",
    "pred_engine.comun.modelos.modelos_fundacionales.chronos2",
)


def _archivos(paquete) -> list[Path]:
    return sorted(Path(paquete.__file__).parent.glob("*.py"))


def _importados(ruta: Path) -> set[str]:
    modulos: set[str] = set()
    for nodo in ast.walk(ast.parse(ruta.read_text(encoding="utf-8"))):
        if isinstance(nodo, ast.Import):
            modulos.update(a.name for a in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            modulos.add(nodo.module)
    return modulos


def _prohibido(modulo: str, prefijo: str) -> bool:
    return modulo == prefijo or modulo.startswith(prefijo + ".")


def test_la_estrategia_no_importa_hpo_walk_forward_ni_el_modelo():
    for ruta in _archivos(paquete_sel):
        for modulo in _importados(ruta):
            for prefijo in _PROHIBIDOS_EN_LA_ESTRATEGIA:
                assert not _prohibido(modulo, prefijo), (ruta, modulo)


def test_el_modelo_no_depende_de_optimizacion():
    # El pronosticador vive en `comun/`: lo consume el Modulo 3 sin HPO ni router.
    for ruta in _archivos(paquete_modelo):
        for modulo in _importados(ruta):
            assert modulo.split(".")[0] != "optuna", ruta
            assert not _prohibido(modulo, "pred_engine.optimizacion"), ruta


@pytest.mark.parametrize("paquete", [paquete_sel, paquete_modelo])
def test_la_familia_fundacional_no_tiene_bucle_ask_tell(paquete):
    for ruta in _archivos(paquete):
        llamadas = {
            n.func.attr
            for n in ast.walk(ast.parse(ruta.read_text(encoding="utf-8")))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert not (llamadas & _METODOS_DE_BUCLE_ASK_TELL), ruta


def test_solo_pipeline_py_toca_torch_y_chronos():
    for ruta in _archivos(paquete_modelo):
        if ruta.name == "pipeline.py":
            continue
        raices = {m.split(".")[0] for m in _importados(ruta)}
        assert not (raices & {"torch", "chronos"}), ruta


def test_el_router_no_conoce_la_familia_fundacional():
    for ruta in _archivos(paquete_router):
        for modulo in _importados(ruta):
            assert modulo.split(".")[0] not in {"torch", "chronos"}, ruta
            assert "optimizadores" not in modulo, ruta
            assert "modelos_fundacionales" not in modulo, ruta
