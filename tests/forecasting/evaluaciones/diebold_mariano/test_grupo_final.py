"""El ejemplo del grupo final de corridas (guia 3.4, ADR-023) sigue su regla."""

from __future__ import annotations

from pathlib import Path

import pytest

from pred_engine.forecasting.evaluaciones.diebold_mariano import (
    PruebaDM,
    corregir_multiplicidad,
)

GUIA = (
    Path(__file__).parents[4]
    / "docs/features/3.Evaluacion/3.4-evaluacion-retrospectiva/README.md"
)
SIN_P = "sin p"
COLUMNAS = [
    "Categoria",
    "Producto",
    "p semilla 0",
    "p semilla 1",
    "p semilla 2",
    "Signos",
    "p del grupo",
    "p BH",
    "Resultado",
]


def _ejemplo(guia: Path = GUIA) -> list[list[str]]:
    seccion = guia.read_text(encoding="utf-8").split("### Grupo final de corridas")[1]
    filas = [
        [celda.strip() for celda in linea.strip().strip("|").split("|")]
        for linea in seccion.split("\n### ", 1)[0].splitlines()
        if linea.startswith("| ") and not linea.startswith("| ---")
    ]
    cabecera, *datos = filas
    assert cabecera == COLUMNAS
    return datos


def test_el_ejemplo_del_grupo_final_sigue_la_regla_documentada() -> None:
    filas = _ejemplo()
    familias: dict[str, dict[str, PruebaDM]] = {}
    signos: dict[str, list[str]] = {}
    for categoria, producto, *p_corridas, signo, p_grupo, _, resultado in filas:
        if SIN_P in p_corridas:
            # Sin p en alguna corrida el producto no entra en la familia de BH.
            assert (p_grupo, resultado) == (SIN_P, "fuera de la familia")
            continue
        assert float(p_grupo) == max(float(p) for p in p_corridas)
        familias.setdefault(categoria, {})[producto] = PruebaDM(
            30, 7, estadistico=-1.0, p_valor=float(p_grupo)
        )
        signos[producto] = signo.split(", ")
    esperados: dict[str, tuple[float | None, str]] = {}
    for pruebas in familias.values():
        # Un BH por categoria sobre los p del grupo.
        for producto, prueba in corregir_multiplicidad(pruebas, alfa=0.05).items():
            if not prueba.significativa:
                resultado = "no rechazado"
            elif signos[producto] == ["-"] * 3:
                resultado = "mejor"
            else:
                assert set(signos[producto]) == {"-", "+"}
                resultado = "rechazado con signos mixtos"
            esperados[producto] = (prueba.p_ajustado, resultado)
    for _, producto, *_, p_bh, resultado in filas:
        if producto in esperados:
            p_ajustado, esperado = esperados[producto]
            assert float(p_bh) == pytest.approx(p_ajustado)
            assert resultado == esperado
    # El ejemplo cubre un producto sin p, uno con signos mixtos y dos categorias.
    assert {f[-1] for f in filas} >= {
        "mejor",
        "rechazado con signos mixtos",
        "fuera de la familia",
    }
    assert set(familias) == {"smooth", "lumpy"}
