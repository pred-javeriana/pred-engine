"""0.3-B2/B3 - Pruebas del bucle de rechazo y remuestreo."""

from __future__ import annotations

import numpy as np
import pytest

from pred_engine.aumentacion.errores import DivergenceRejectionExhausted
from pred_engine.aumentacion.rechazo import generar_series_aceptadas

_SEMILLA = np.abs(np.random.default_rng(0).normal(50.0, 5.0, 120)).round()


def _generador_perfecto(rng: np.random.Generator) -> np.ndarray:
    """Candidata siempre dentro de tolerancia (ruido minusculo)."""
    return _SEMILLA + rng.normal(0.0, 0.05, _SEMILLA.size)


def _generador_siempre_malo(rng: np.random.Generator) -> np.ndarray:
    """Candidata con la media desplazada muy por encima de la tolerancia."""
    return _SEMILLA * 2.0 + rng.normal(0.0, 1.0, _SEMILLA.size)


def test_produce_el_numero_de_series_solicitado() -> None:
    resultado = generar_series_aceptadas(
        _SEMILLA,
        period=7,
        n_series=4,
        generador=_generador_perfecto,
    )
    assert len(resultado.series) == 4
    assert resultado.tasa_rechazo == 0.0


def test_es_reproducible_con_semilla_fija() -> None:
    kwargs = dict(period=7, n_series=3, generador=_generador_perfecto)
    a = generar_series_aceptadas(_SEMILLA, semilla_aleatoria=99, **kwargs)
    b = generar_series_aceptadas(_SEMILLA, semilla_aleatoria=99, **kwargs)
    assert all(np.array_equal(x, y) for x, y in zip(a.series, b.series))


def test_semillas_distintas_producen_paneles_distintos() -> None:
    kwargs = dict(period=7, n_series=2, generador=_generador_perfecto)
    a = generar_series_aceptadas(_SEMILLA, semilla_aleatoria=1, **kwargs)
    b = generar_series_aceptadas(_SEMILLA, semilla_aleatoria=2, **kwargs)
    assert not np.array_equal(a.series[0], b.series[0])


def test_agotar_reintentos_eleva_excepcion_controlada() -> None:
    with pytest.raises(DivergenceRejectionExhausted):
        generar_series_aceptadas(
            _SEMILLA,
            period=7,
            n_series=1,
            max_reintentos=3,
            generador=_generador_siempre_malo,
        )


def test_registra_la_tasa_de_rechazo_global() -> None:
    estado = {"llamadas": 0}

    def generador_intermitente(rng: np.random.Generator) -> np.ndarray:
        estado["llamadas"] += 1
        if estado["llamadas"] % 2 == 1:
            return _generador_siempre_malo(rng)
        return _generador_perfecto(rng)

    resultado = generar_series_aceptadas(
        _SEMILLA,
        period=7,
        n_series=2,
        max_reintentos=5,
        generador=generador_intermitente,
    )
    assert resultado.rechazos == 2
    assert resultado.intentos == 4
    assert resultado.tasa_rechazo == pytest.approx(0.5)


def test_parametros_invalidos_son_rechazados() -> None:
    with pytest.raises(ValueError):
        generar_series_aceptadas(_SEMILLA, period=7, n_series=0)
    with pytest.raises(ValueError):
        generar_series_aceptadas(_SEMILLA, period=7, n_series=1, max_reintentos=0)


def test_motor_mbb_por_defecto_reutiliza_las_primitivas_del_companiero() -> None:
    """Sin ``generador`` inyectado usa STL + MBB de ``mbb.py``."""
    base = np.arange(140)
    semilla = (
        30.0
        + 0.05 * base
        + 6.0 * np.sin(2 * np.pi * base / 7)
        + np.random.default_rng(3).normal(0.0, 1.0, base.size)
    ).clip(min=0)
    resultado = generar_series_aceptadas(
        semilla,
        period=7,
        n_series=2,
        tolerancia=0.2,
        max_reintentos=50,
        semilla_aleatoria=5,
    )
    assert len(resultado.series) == 2
    assert all(s.shape == semilla.shape for s in resultado.series)
