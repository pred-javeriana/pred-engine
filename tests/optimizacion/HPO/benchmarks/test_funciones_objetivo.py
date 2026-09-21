"""Funciones objetivo sinteticas: minimo global conocido en el origen (salvo
Rosenbrock, en (1,...,1))."""

from __future__ import annotations

import pytest

from pred_engine.optimizacion.optimizadores.HPO.benchmarks.funciones_objetivo import (
    rastrigin,
    rosenbrock,
    sphere,
)


def test_sphere_es_cero_en_el_origen():
    assert sphere({"x0": 0.0, "x1": 0.0, "x2": 0.0}) == 0.0


def test_sphere_crece_con_la_distancia_al_origen():
    assert sphere({"x0": 1.0}) < sphere({"x0": 2.0})


def test_rastrigin_es_cero_en_el_origen():
    assert rastrigin({"x0": 0.0, "x1": 0.0}) == pytest.approx(0.0, abs=1e-9)


def test_rastrigin_es_multimodal_alrededor_del_origen():
    # A diferencia de sphere, un punto ligeramente mas lejos puede valer
    # MENOS que uno mas cercano -- esa es la propiedad que la hace un
    # benchmark mas dificil para busqueda aleatoria/TPE.
    valores = [rastrigin({"x0": x}) for x in (0.5, 1.0, 1.5, 2.0)]
    assert not all(a <= b for a, b in zip(valores, valores[1:]))


def test_rosenbrock_es_cero_en_el_optimo_conocido():
    assert rosenbrock({"x0": 1.0, "x1": 1.0, "x2": 1.0}) == pytest.approx(0.0, abs=1e-9)


def test_rosenbrock_exige_al_menos_dos_dimensiones():
    with pytest.raises(ValueError):
        rosenbrock({"x0": 1.0})
