"""Unidades aisladas: un fallo no detiene a las demas, y el paralelismo es real."""

from __future__ import annotations

import math
import os
import time

import pytest

from pred_engine.comun.ejecucion_paralela import (
    RegistroUnidad,
    ejecutar_unidades,
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
