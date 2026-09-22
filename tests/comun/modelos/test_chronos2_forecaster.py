"""Chronos2Forecaster: contrato de Pronosticador, configuracion congelada y fail-closed.

Usa `PipelineFalso` (sin torch): el modelo real se prueba en
`test_chronos2_real.py` (slow, requiere el extra `foundation`).
"""

from __future__ import annotations

import builtins
import dataclasses

import numpy as np
import pytest
from tests._dobles import PipelineFalso, PipelineQueFalla

from pred_engine.comun.modelos.modelos_fundacionales import (
    CHRONOS2_ZERO_SHOT,
    AjusteModeloError,
    Chronos2Forecaster,
    ModeloFundacionalNoDisponibleError,
    cargar_pipeline,
    fabrica_fundacional,
)
from pred_engine.comun.walkforward import evaluar_walk_forward
from pred_engine.comun.walkforward.protocolos import Pronosticador


def _serie(n: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.maximum(0.0, 10 + 5 * np.sin(np.arange(n) * 2 * np.pi / 7)) + rng.normal(
        0, 1, n
    ).clip(0, 3)


def _modelo(pipeline=None) -> Chronos2Forecaster:
    return Chronos2Forecaster(pipeline=pipeline or PipelineFalso())


def test_satisface_el_contrato_pronosticador():
    assert isinstance(_modelo(), Pronosticador)


def test_fit_devuelve_self_y_predict_tiene_la_longitud_del_horizonte():
    modelo = _modelo()
    assert modelo.fit(_serie()) is modelo
    assert modelo.predict(14).shape == (14,)


def test_predict_antes_de_fit_falla():
    with pytest.raises(RuntimeError, match="fit"):
        _modelo().predict(3)


def test_horizonte_invalido_falla():
    with pytest.raises(ValueError, match="horizon"):
        _modelo().fit(_serie()).predict(0)


@pytest.mark.parametrize("y", [np.array([]), np.array([1.0, np.nan]), np.ones((2, 3))])
def test_serie_invalida_falla(y):
    with pytest.raises(ValueError):
        _modelo().fit(y)


def test_el_pronostico_puntual_es_la_mediana():
    y = _serie()
    pronostico = _modelo().fit(y).predict(5)
    np.testing.assert_array_equal(pronostico, np.full(5, y[-1]))


def test_predict_cuantiles_devuelve_todos_los_niveles_recortados_a_cero():
    modelo = _modelo().fit(np.array([0.0, 2.0, 1.0]))
    cuantiles = modelo.predict_cuantiles(4)
    assert cuantiles.shape == (3, 4)
    # El cuantil 0.1 del doble es ultimo - 5 = -4: la demanda no es negativa.
    np.testing.assert_array_equal(cuantiles[0], np.zeros(4))
    np.testing.assert_array_equal(cuantiles[2], np.full(4, 2.0))
    assert modelo.cuantiles == (0.1, 0.5, 0.9)


def test_el_contexto_se_recorta_al_limite_nativo_del_modelo():
    pipeline = PipelineFalso()
    largo = CHRONOS2_ZERO_SHOT.max_contexto + 100
    y = np.arange(largo, dtype=float)
    _modelo(pipeline).fit(y).predict(1)
    limite = CHRONOS2_ZERO_SHOT.max_contexto
    np.testing.assert_array_equal(pipeline.contextos[0], y[-limite:])


def test_el_modelo_solo_ve_lo_que_recibe_en_fit():
    pipeline = PipelineFalso()
    y = _serie()
    modelo = _modelo(pipeline).fit(y)
    y[-1] = 999.0  # mutar el arreglo original despues de fit no fuga al modelo
    modelo.predict(3)
    assert pipeline.contextos[0][-1] != 999.0


def test_falla_de_inferencia_se_traduce_a_ajuste_modelo_error():
    with pytest.raises(AjusteModeloError, match="inferencia rota"):
        _modelo(PipelineQueFalla()).fit(_serie()).predict(3)


def test_forma_inesperada_de_cuantiles_falla_cerrado():
    class _PipelineForma(PipelineFalso):
        def pronosticar_cuantiles(self, contexto, horizonte):
            return np.zeros((2, horizonte))

    with pytest.raises(AjusteModeloError, match="forma"):
        _modelo(_PipelineForma()).fit(_serie()).predict(3)


def test_modelo_sin_el_cuantil_puntual_falla_cerrado():
    sin_mediana = PipelineFalso(cuantiles=(0.1, 0.9), desplazamientos=(0.0, 1.0))
    with pytest.raises(AjusteModeloError, match="0.5"):
        _modelo(sin_mediana).fit(_serie()).predict(3)


def test_mismo_contexto_mismo_pronostico():
    y = _serie()
    a = _modelo().fit(y).predict(7)
    b = _modelo().fit(y).predict(7)
    np.testing.assert_array_equal(a, b)


def test_la_configuracion_es_inmutable_y_unica():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CHRONOS2_ZERO_SHOT.revision = "otra"  # type: ignore[misc]
    assert _modelo().configuracion is CHRONOS2_ZERO_SHOT


def test_descripcion_canonica_fija_modelo_revision_y_modo_zero_shot():
    descripcion = CHRONOS2_ZERO_SHOT.descripcion_canonica()
    assert descripcion["model_id"] == "amazon/chronos-2"
    assert len(descripcion["revision"]) == 40
    assert descripcion["cuantil_puntual"] == 0.5
    assert descripcion["aprendizaje_cruzado"] is False
    assert descripcion == CHRONOS2_ZERO_SHOT.descripcion_canonica()


def test_la_fabrica_rechaza_cualquier_configuracion():
    with pytest.raises(ValueError, match="no se configura"):
        fabrica_fundacional({"max_contexto": 512}, seed=0, pipeline=PipelineFalso())


def test_la_fabrica_se_enchufa_al_walk_forward_existente():
    pipeline = PipelineFalso()

    def _fabrica(configuracion, *, seed):
        return fabrica_fundacional(configuracion, seed=seed, pipeline=pipeline)

    resultado = evaluar_walk_forward(
        _serie(), _fabrica, {}, min_train=28, horizonte=7, paso=7
    )
    assert resultado.n_ventanas_evaluadas == resultado.n_ventanas_totales > 0
    # Cada ventana recibe exactamente su tramo de entrenamiento (ventana expansiva).
    assert [len(c) for c in pipeline.contextos] == [
        v.ventana.n_train for v in resultado.ventanas
    ]


def test_sin_el_extra_foundation_falla_con_error_explicito(monkeypatch):
    importar = builtins.__import__

    def _sin_chronos(nombre, *args, **kwargs):
        if nombre in ("chronos", "torch"):
            raise ImportError(nombre)
        return importar(nombre, *args, **kwargs)

    cargar_pipeline.cache_clear()
    monkeypatch.setattr(builtins, "__import__", _sin_chronos)
    try:
        with pytest.raises(ModeloFundacionalNoDisponibleError, match="foundation"):
            Chronos2Forecaster().fit(_serie()).predict(3)
    finally:
        cargar_pipeline.cache_clear()
