"""Flujo de progreso de las unidades (G3): una linea JSON por evento, mientras corre."""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
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


def _la_primera_espera_la_senal(tarea: tuple[int, str]) -> int:
    indice, senal = tarea
    limite = time.monotonic() + 30
    while indice == 0 and not Path(senal).exists():
        if time.monotonic() > limite:
            raise TimeoutError("la senal no llego")
        time.sleep(0.01)
    return indice


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


def _emitir_muchos(ruta: str, escritor: int, n: int) -> None:
    for i in range(n):
        emitir_evento(ruta, "unit_finished", escritor=escritor, i=i, relleno="x" * 512)


def test_varios_procesos_no_intercalan_lineas(tmp_path: Path) -> None:
    """Cada evento queda en una linea JSON entera aunque escriban varios procesos."""
    ruta = tmp_path / "progreso.jsonl"
    contexto = multiprocessing.get_context("spawn")
    procesos = [
        contexto.Process(target=_emitir_muchos, args=(str(ruta), escritor, 300))
        for escritor in range(4)
    ]
    for proceso in procesos:
        proceso.start()
    for proceso in procesos:
        proceso.join(timeout=120)
        assert proceso.exitcode == 0

    eventos = _eventos(ruta)
    assert sorted((e["escritor"], e["i"]) for e in eventos) == [
        (escritor, i) for escritor in range(4) for i in range(300)
    ]


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


def test_el_fin_de_una_unidad_se_escribe_mientras_otra_sigue_corriendo(
    tmp_path: Path,
) -> None:
    ruta, senal = tmp_path / "progreso.jsonl", tmp_path / "senal"
    corrida = threading.Thread(
        target=ejecutar_unidades,
        args=(_la_primera_espera_la_senal, [(0, str(senal)), (1, str(senal))]),
        kwargs={"procesos": 2, "etiquetas": _etiquetas(2), "progreso": ruta},
    )

    def terminadas() -> set[str]:
        lineas = ruta.read_text(encoding="utf-8").splitlines() if ruta.exists() else []
        eventos = [json.loads(linea) for linea in lineas if linea.endswith("}")]
        return {e["sku_id"] for e in eventos if e["event"] == "unit_finished"}

    corrida.start()
    try:
        limite = time.monotonic() + 20
        while "SKU-1" not in terminadas() and time.monotonic() < limite:
            time.sleep(0.05)
        # SKU-1 ya termino y esta escrito aunque SKU-0, antes en la lista, siga.
        assert terminadas() == {"SKU-1"}
    finally:
        senal.touch()
        corrida.join(timeout=30)
    assert terminadas() == {"SKU-0", "SKU-1"}


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


def test_la_unidad_fallida_lleva_su_codigo_en_el_registro_y_en_el_evento(
    tmp_path: Path,
) -> None:
    ruta = tmp_path / "progreso.jsonl"
    etiquetas = [
        EtiquetaUnidad(etapa, f"SKU-{i}", "classical", "sarima")
        for i, etapa in enumerate(["L2", "L3", "L2"])
    ]

    resultados = ejecutar_unidades(
        _falla_en_tres, [1, 3, 3], etiquetas=etiquetas, progreso=ruta
    )

    assert resultados[0].registro.codigo is None
    # Sin tipo propio: el respaldo depende de la etapa de la unidad.
    assert [r.registro.codigo for r in resultados[1:]] == [
        "evaluation.failed",
        "pipeline.stage_failed",
    ]
    fines = {e["sku_id"]: e for e in _eventos(ruta) if e["event"] == "unit_finished"}
    assert fines["SKU-0"]["code"] is None
    assert fines["SKU-1"]["code"] == "evaluation.failed"
    assert fines["SKU-2"]["code"] == "pipeline.stage_failed"


def test_el_codigo_viaja_desde_el_proceso_hijo(tmp_path: Path) -> None:
    ruta = tmp_path / "progreso.jsonl"

    resultados = ejecutar_unidades(
        _falla_en_tres, [3, 1], procesos=2, etiquetas=_etiquetas(2), progreso=ruta
    )

    assert resultados[0].registro.codigo == "pipeline.stage_failed"
    assert resultados[1].registro.codigo is None
