"""Flujo de progreso de las unidades (G3): una linea JSON por evento, mientras corre."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from pred_engine.comun.ejecucion_paralela import (
    EtiquetaUnidad,
    ejecutar_unidades,
    emitir_evento,
)


def _doble(x: int) -> int:
    return x * 2


def _falla_en_tres(x: int) -> int:
    if x == 3:
        raise ValueError("tres no vale")
    return x


def _mata_al_proceso(x: int) -> int:
    if x == 1:
        os._exit(7)  # el hijo muere sin devolver nada
    return x


def _eventos(ruta: Path) -> list[dict[str, Any]]:
    return [
        json.loads(linea) for linea in ruta.read_text(encoding="utf-8").splitlines()
    ]


def _etiquetas(n: int) -> list[EtiquetaUnidad]:
    return [EtiquetaUnidad("L2", f"SKU-{i}", "classical", "sarima") for i in range(n)]


def test_emitir_evento_agrega_una_linea_json(tmp_path: Path) -> None:
    ruta = tmp_path / "progreso.jsonl"
    emitir_evento(ruta, "stage_started", stage="L1")
    emitir_evento(ruta, "stage_finished", stage="L1", state="completed")

    primero, segundo = _eventos(ruta)
    assert (primero["event"], primero["stage"]) == ("stage_started", "L1")
    assert segundo["state"] == "completed"
    assert primero["t"].endswith("+00:00")


def test_un_fallo_al_escribir_no_interrumpe(tmp_path: Path) -> None:
    emitir_evento(
        tmp_path, "stage_started", stage="L1"
    )  # es un directorio: no se puede abrir


def test_secuencial_anota_inicio_y_fin_de_cada_unidad(tmp_path: Path) -> None:
    ruta = tmp_path / "progreso.jsonl"

    resultados = ejecutar_unidades(
        _doble, [1, 2, 3], etiquetas=_etiquetas(3), progreso=ruta
    )

    assert [r.valor for r in resultados] == [2, 4, 6]
    eventos = _eventos(ruta)
    assert [e["event"] for e in eventos] == ["unit_started", "unit_finished"] * 3
    fin = eventos[1]
    assert fin["stage"] == "L2" and fin["sku_id"] == "SKU-0"
    assert fin["family"] == "classical" and fin["model"] == "sarima"
    assert fin["state"] == "completada" and fin["error"] is None
    assert fin["seconds"] >= 0 and fin["pid"] == os.getpid()


def test_una_unidad_fallida_anota_su_error_y_las_demas_siguen(tmp_path: Path) -> None:
    ruta = tmp_path / "progreso.jsonl"

    ejecutar_unidades(_falla_en_tres, [1, 3, 5], etiquetas=_etiquetas(3), progreso=ruta)

    fines = [e for e in _eventos(ruta) if e["event"] == "unit_finished"]
    assert [(e["sku_id"], e["state"]) for e in fines] == [
        ("SKU-0", "completada"),
        ("SKU-1", "fallida"),
        ("SKU-2", "completada"),
    ]
    assert fines[1]["error"] == "ValueError: tres no vale"


def test_en_paralelo_cada_hijo_anota_su_unidad(tmp_path: Path) -> None:
    ruta = tmp_path / "progreso.jsonl"

    ejecutar_unidades(
        _doble, list(range(4)), procesos=2, etiquetas=_etiquetas(4), progreso=ruta
    )

    eventos = _eventos(ruta)
    for i in range(4):
        propios = [e for e in eventos if e["sku_id"] == f"SKU-{i}"]
        assert [e["event"] for e in propios] == ["unit_started", "unit_finished"]
        assert propios[1]["state"] == "completada"
    assert {e["pid"] for e in eventos} != {os.getpid()}


def test_si_el_hijo_muere_el_padre_anota_el_fin_fallido(tmp_path: Path) -> None:
    ruta = tmp_path / "progreso.jsonl"

    resultados = ejecutar_unidades(
        _mata_al_proceso, [0, 1], procesos=2, etiquetas=_etiquetas(2), progreso=ruta
    )

    assert [r.registro.estado for r in resultados].count("fallida") >= 1
    fines = [e for e in _eventos(ruta) if e["event"] == "unit_finished"]
    assert {e["sku_id"] for e in fines} == {"SKU-0", "SKU-1"} or any(
        e["state"] == "fallida" and e["error"] for e in fines
    )
    assert all(e["state"] == "fallida" for e in fines if e["sku_id"] == "SKU-1")


def test_sin_progreso_no_se_escribe_nada_y_el_resultado_es_igual(
    tmp_path: Path,
) -> None:
    resultados = ejecutar_unidades(_doble, [1, 2], etiquetas=_etiquetas(2))
    assert [r.valor for r in resultados] == [2, 4]
    assert list(tmp_path.iterdir()) == []


def test_las_etiquetas_deben_ser_una_por_tarea(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="una entrada por tarea"):
        ejecutar_unidades(
            _doble, [1, 2], etiquetas=_etiquetas(1), progreso=tmp_path / "p.jsonl"
        )
