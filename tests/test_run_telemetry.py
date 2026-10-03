"""Muestreo de recursos: procesos reales del pool, costo propio y fallos aislados."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from pred_engine.comun.ejecucion_paralela import ejecutar_unidades
from pred_engine.run_telemetry import ResourceSampler


def _ocupar(segundos: float) -> int:
    """CPU real en el proceso hijo (nivel de modulo para ``spawn``)."""
    fin, vueltas = time.perf_counter() + segundos, 0
    while time.perf_counter() < fin:
        vueltas += 1
    return vueltas


def _filas(ruta: Path) -> list[dict]:
    return [json.loads(linea) for linea in ruta.read_text().splitlines()]


def test_registra_cpu_y_memoria_del_proceso_principal_y_de_cada_worker(tmp_path):
    ruta = tmp_path / "corrida" / "recursos.jsonl"
    with ResourceSampler(ruta, interval=0.1) as sampler:
        resultados = ejecutar_unidades(_ocupar, [1.0, 1.0], procesos=2)
    assert all(r.registro.estado == "completada" for r in resultados)

    filas = _filas(ruta)
    assert filas[0]["role"] == "host"
    assert filas[0]["cpu_cores"] == os.cpu_count()
    assert filas[0]["mem_mb"] > 0
    assert set(filas[0]) == {"at", "role", "pid", "cpu_cores", "mem_mb"}
    assert all(set(f) == set(filas[0]) for f in filas)
    assert {f["pid"] for f in filas if f["role"] == "main"} == {os.getpid()}
    assert any(f["role"] == "system" for f in filas)

    workers = [f for f in filas if f["role"] == "worker"]
    pids = {r.registro.pid for r in resultados}
    assert pids <= {f["pid"] for f in workers}
    # Cada worker ocupo un nucleo durante su unidad: alguna muestra lo muestra.
    for pid in pids:
        assert max(f["cpu_cores"] for f in workers if f["pid"] == pid) > 0.5
        assert all(f["mem_mb"] > 0 for f in workers if f["pid"] == pid)

    resumen = sampler.summary()
    assert resumen["file"] == "recursos.jsonl"
    assert resumen["samples"] >= 5
    assert resumen["peak_workers"] == 2
    assert resumen["peak_cpu_cores"] > 1
    assert resumen["peak_rss_mb"] > 0
    assert 0 <= resumen["sampler_cpu_s"] < 1
    assert resumen["sampler_ms_per_sample"] > 0
    assert resumen["error"] is None


def test_un_error_de_muestreo_no_interrumpe_la_corrida(tmp_path, monkeypatch):
    sampler = ResourceSampler(tmp_path / "recursos.jsonl", interval=0.05)

    def falla():
        raise RuntimeError("proc no disponible")

    monkeypatch.setattr(sampler, "_filas", falla)
    with sampler:
        time.sleep(0.2)
    resumen = sampler.summary()
    assert resumen["error"] == "RuntimeError: proc no disponible"
    assert [f["role"] for f in _filas(tmp_path / "recursos.jsonl")] == ["host"]


def test_reescribe_el_archivo_de_una_ejecucion_anterior(tmp_path):
    ruta = tmp_path / "recursos.jsonl"
    ruta.write_text('{"viejo": true}\n')
    with ResourceSampler(ruta, interval=10):
        pass
    filas = _filas(ruta)
    assert filas[0]["role"] == "host"
    assert all("viejo" not in f for f in filas)
    # Al detenerse siempre toma una ultima muestra, aunque no llegue el intervalo.
    assert any(f["role"] == "main" for f in filas)


@pytest.mark.parametrize("interval", [0, -1.0])
def test_rechaza_intervalos_no_positivos(tmp_path, interval):
    with pytest.raises(ValueError, match="interval"):
        ResourceSampler(tmp_path / "r.jsonl", interval=interval)


def test_no_arranca_dos_veces_y_detener_sin_arrancar_no_hace_nada(tmp_path):
    sampler = ResourceSampler(tmp_path / "r.jsonl", interval=10)
    sampler.stop()
    assert not (tmp_path / "r.jsonl").exists()
    with sampler, pytest.raises(RuntimeError, match="already started"):
        sampler.start()
