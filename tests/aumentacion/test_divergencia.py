"""0.3-B1 - Pruebas del calculador de divergencia parametrica."""

from __future__ import annotations

import numpy as np
import pytest

from pred_engine.aumentacion.divergencia import evaluar_divergencia_parametrica


def test_serie_identica_es_aceptada_y_reporta_estadisticos() -> None:
    semilla = np.array([10.0, 12.0, 8.0, 11.0, 9.0])
    veredicto = evaluar_divergencia_parametrica(semilla, semilla.copy())
    assert veredicto.aceptada
    assert veredicto.divergencia_media == pytest.approx(0.0)
    assert veredicto.divergencia_varianza == pytest.approx(0.0)
    assert veredicto.media_semilla == pytest.approx(veredicto.media_candidata)


def test_divergencia_de_media_supera_tolerancia() -> None:
    semilla = np.full(50, 10.0)
    candidata = np.full(50, 10.6)  # +6 % en media
    veredicto = evaluar_divergencia_parametrica(semilla, candidata, tolerancia=0.05)
    assert not veredicto.aceptada
    assert veredicto.divergencia_media == pytest.approx(0.06)


def test_tolerancia_es_configurable() -> None:
    semilla = np.full(50, 10.0)
    candidata = np.full(50, 10.6)
    assert evaluar_divergencia_parametrica(semilla, candidata, tolerancia=0.10).aceptada


def test_tolerancia_no_positiva_es_rechazada() -> None:
    with pytest.raises(ValueError, match="tolerancia"):
        evaluar_divergencia_parametrica(np.ones(3), np.ones(3), tolerancia=0.0)


def test_series_vacias_son_rechazadas() -> None:
    with pytest.raises(ValueError):
        evaluar_divergencia_parametrica(np.array([]), np.ones(3))
