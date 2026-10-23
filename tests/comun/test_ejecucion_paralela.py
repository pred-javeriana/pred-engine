"""Unidades aisladas: un fallo no detiene a las demas, y el paralelismo es real."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path

import psutil
import pytest

from pred_engine.comun.ejecucion_paralela import (
    RegistroUnidad,
    ejecutar_unidades,
    iterar_unidades,
    resumen_paralelismo,
)


@pytest.mark.parametrize("procesos", [1, 2])
def test_una_unidad_fallida_no_detiene_a_las_demas(procesos):
    resultados = ejecutar_unidades(math.sqrt, [4.0, -1.0, 9.0], procesos=procesos)

    assert [r.valor for r in resultados] == [2.0, None, 3.0]
    assert [r.registro.estado for r in resultados] == [
        "completada",
        "fallida",
        "completada",
    ]
    assert resultados[1].registro.error == "ValueError: math domain error"
    assert all(r.registro.fin >= r.registro.inicio for r in resultados)


def test_modo_secuencial_corre_en_el_proceso_actual():
    resultados = ejecutar_unidades(math.sqrt, [1.0, 16.0])
    assert {r.registro.pid for r in resultados} == {os.getpid()}


def test_modo_paralelo_solapa_unidades_en_procesos_distintos():
    resultados = ejecutar_unidades(time.sleep, [0.6] * 4, procesos=4)

    registros = [r.registro for r in resultados]
    resumen = resumen_paralelismo(registros)
    assert all(r.estado == "completada" for r in registros)
    assert os.getpid() not in {r.pid for r in registros}
    assert resumen["procesos_distintos"] >= 2
    assert resumen["concurrencia_maxima"] >= 2
    assert resumen["aceleracion"] > 1.5
    assert sum(r.duracion_s for r in registros) > 2.3


def test_resumen_cuenta_solapes_y_no_tramos_contiguos():
    registros = [
        RegistroUnidad("completada", 1, 0.0, 2.0),
        RegistroUnidad("completada", 2, 1.0, 3.0),
        RegistroUnidad("fallida", 1, 3.0, 4.0, "ValueError: x"),
    ]
    resumen = resumen_paralelismo(registros)
    assert resumen["concurrencia_maxima"] == 2
    assert resumen["procesos_distintos"] == 2
    assert resumen["computo_s"] == 5.0
    assert resumen["pared_s"] == 4.0
    assert resumen["aceleracion"] == 1.25
    assert resumen_paralelismo([])["unidades"] == 0


@pytest.mark.parametrize("procesos", [0, -1, 1.5, True])
def test_rechaza_cantidad_de_procesos_invalida(procesos):
    with pytest.raises(ValueError, match="procesos"):
        ejecutar_unidades(math.sqrt, [1.0], procesos=procesos)


def _ocupar(segundos: float) -> None:
    fin = time.process_time() + segundos
    while time.process_time() < fin:
        pass


@pytest.mark.parametrize("procesos", [1, 2])
def test_cada_unidad_registra_su_tiempo_de_cpu(procesos):
    resultados = ejecutar_unidades(_ocupar, [0.3, 0.3], procesos=procesos)
    for resultado in resultados:
        registro = resultado.registro
        assert registro.cpu_s is not None
        assert 0.2 < registro.cpu_s <= registro.duracion_s + 0.1


def _cuadrados(n: int, leidas: list[int]) -> Iterator[float]:
    for i in range(n):
        leidas.append(i)
        yield float(i * i)


@pytest.mark.parametrize("procesos", [1, 2])
def test_iterar_entrega_en_orden_y_lee_las_tareas_de_a_poco(procesos):
    leidas: list[int] = []
    resultados = iterar_unidades(
        math.sqrt, _cuadrados(8, leidas), procesos=procesos, en_vuelo=3
    )

    assert next(resultados).valor == 0.0
    assert len(leidas) <= 3
    assert [r.valor for r in resultados] == [float(i) for i in range(1, 8)]
    assert len(leidas) == 8


@pytest.mark.parametrize("procesos", [1, 2])
def test_iterar_aisla_una_unidad_fallida(procesos):
    resultados = list(
        iterar_unidades(math.sqrt, iter([4.0, -1.0, 9.0]), procesos=procesos)
    )
    assert [r.valor for r in resultados] == [2.0, None, 3.0]
    assert resultados[1].registro.error == "ValueError: math domain error"


def _marcar(tarea: tuple[Path, int]) -> int:
    carpeta, i = tarea
    (carpeta / str(i)).touch()
    time.sleep(0.3)
    return i


def test_dejar_de_consumir_cancela_las_tareas_no_enviadas(tmp_path):
    resultados = iterar_unidades(
        _marcar, [(tmp_path, i) for i in range(12)], procesos=2, en_vuelo=2
    )
    assert next(resultados).valor == 0
    resultados.close()
    assert len(list(tmp_path.iterdir())) <= 3


@pytest.mark.parametrize("en_vuelo", [0, -1, 1.5])
def test_iterar_rechaza_en_vuelo_invalido(en_vuelo):
    with pytest.raises(ValueError, match="en_vuelo"):
        iterar_unidades(math.sqrt, [1.0], procesos=2, en_vuelo=en_vuelo)


_PADRE_QUE_MUERE = textwrap.dedent(
    """
    import os
    import signal
    import sys
    import threading
    import time
    from pathlib import Path

    from pred_engine.comun.ejecucion_paralela import iterar_unidades


    def anotar(carpeta):
        Path(carpeta, str(os.getpid())).touch()
        time.sleep(120)


    if __name__ == "__main__":
        carpeta = Path(sys.argv[1])
        resultados = iterar_unidades(anotar, [str(carpeta)] * 2, procesos=2)
        threading.Thread(target=lambda: next(resultados), daemon=True).start()
        while len(list(carpeta.iterdir())) < 2:
            time.sleep(0.1)
        os.kill(os.getpid(), signal.SIGKILL)
    """
)


def _vivo(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_los_procesos_del_pool_terminan_si_el_padre_muere(tmp_path):
    guion = tmp_path / "padre.py"
    guion.write_text(_PADRE_QUE_MUERE)
    carpeta = tmp_path / "hijos"
    carpeta.mkdir()
    padre = subprocess.run(
        [sys.executable, str(guion), str(carpeta)], timeout=120, check=False
    )
    assert padre.returncode == -9
    hijos = [int(p.name) for p in carpeta.iterdir()]
    assert len(hijos) == 2
    limite = time.monotonic() + 15
    while any(_vivo(pid) for pid in hijos) and time.monotonic() < limite:
        time.sleep(0.2)
    assert not any(_vivo(pid) for pid in hijos)
