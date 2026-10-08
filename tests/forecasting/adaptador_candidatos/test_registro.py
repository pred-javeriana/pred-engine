"""Registro 3.2-A1/A2: instanciacion homogenea, sin defaults y reproducible."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import TypeAdapter
from tests._dobles import PipelineFalso
from tests._manifiestos import CONFIGURACIONES, candidato

from pred_engine.comun.modelos.manifiesto_candidatos import Candidato
from pred_engine.comun.walkforward.protocolos import Pronosticador
from pred_engine.forecasting.adaptador_candidatos import (
    FABRICAS,
    PERIODO_LINEA_BASE,
    AdaptadorCandidato,
    ModeloNoRegistradoError,
    instanciar,
    instanciar_linea_base,
)

_FAMILIAS = ["classical", "ml", "dl", "foundation"]

_ADAPTADOR: TypeAdapter[Candidato] = TypeAdapter(Candidato)


def _candidato(familia: str, **cambios: object) -> Candidato:
    return _ADAPTADOR.validate_python(candidato(familia, **cambios))


def _serie(n: int = 60) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(n)
    return np.clip(20 + 5 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1, n), 0, None)


def _pronostico(familia: str, semilla: int = 7) -> np.ndarray:
    modelo = instanciar(
        _candidato(familia, semilla=semilla), pipeline_fundacional=PipelineFalso()
    )
    return modelo.fit(_serie()).predict(7)


def _adaptador(familia: str) -> AdaptadorCandidato:
    if familia == "linea_base":
        return instanciar_linea_base()
    return instanciar(_candidato(familia), pipeline_fundacional=PipelineFalso())


@pytest.mark.parametrize("familia", _FAMILIAS)
def test_cada_familia_cumple_el_contrato_pronosticador(familia: str) -> None:
    modelo = instanciar(_candidato(familia), pipeline_fundacional=PipelineFalso())
    assert isinstance(modelo, Pronosticador)
    assert isinstance(modelo, AdaptadorCandidato)
    assert modelo.seed == 7  # type: ignore[attr-defined]


# --- ADR-03-004: ajustar una vez en t* y pronosticar desde cada origen --------


@pytest.mark.parametrize("familia", [*_FAMILIAS, "linea_base"])
def test_sin_observaciones_nuevas_pronosticar_es_predict(familia: str) -> None:
    modelo = _adaptador(familia).fit(_serie())
    assert np.array_equal(
        modelo.pronosticar(np.empty(0), 7),
        modelo.predict(7),  # type: ignore[attr-defined]
    )


@pytest.mark.parametrize("familia", [*_FAMILIAS, "linea_base"])
def test_pronosticar_usa_lo_observado_sin_reajustar(
    familia: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    modelo = _adaptador(familia).fit(_serie())
    antes = modelo.predict(5)  # type: ignore[attr-defined]

    def _no_reajustar(y: np.ndarray) -> None:
        raise AssertionError("pronosticar no debe volver a ajustar")

    monkeypatch.setattr(modelo, "fit", _no_reajustar)
    # Ceros, lejos del nivel (~20) de la historia: el pronostico debe notarlo.
    observadas = np.zeros(10)
    desde_origen = modelo.pronosticar(observadas, 5)

    assert desde_origen.shape == (5,) and np.all(np.isfinite(desde_origen))
    assert not np.array_equal(desde_origen, antes)
    assert np.array_equal(modelo.pronosticar(observadas, 5), desde_origen)
    # Cada origen se pronostica aparte: lo ajustado no cambia.
    assert np.array_equal(modelo.predict(5), antes)  # type: ignore[attr-defined]


@pytest.mark.parametrize("familia", ["classical", "ml", "dl", "foundation"])
def test_misma_semilla_mismo_pronostico(familia: str) -> None:
    primero = _pronostico(familia)
    assert primero.shape == (7,)
    assert np.all(np.isfinite(primero))
    assert np.array_equal(primero, _pronostico(familia))


def test_la_configuracion_recibida_llega_intacta_al_modelo() -> None:
    sarima = instanciar(_candidato("classical"))
    assert sarima.order == (1, 0, 0)  # type: ignore[attr-defined]
    assert sarima.seasonal_order == (0, 0, 0, 7)  # type: ignore[attr-defined]

    lgbm = instanciar(_candidato("ml"))
    cfg = CONFIGURACIONES["ml"]
    # n_estimators=5 y lags=3 difieren del default del constructor (100 y 7).
    assert lgbm.n_estimators == cfg["n_estimators"]  # type: ignore[attr-defined]
    assert lgbm.lags == cfg["lags"]  # type: ignore[attr-defined]
    assert lgbm.estacionalidad == cfg["m"]  # type: ignore[attr-defined]

    mlp = instanciar(_candidato("dl"))
    for campo, valor in CONFIGURACIONES["dl"].items():
        assert getattr(mlp, campo) == valor


def test_familia_sin_fabrica_no_se_instancia() -> None:
    fabricas = {f: fabrica for f, fabrica in FABRICAS.items() if f != "ml"}
    with pytest.raises(ModeloNoRegistradoError):
        instanciar(_candidato("ml"), fabricas=fabricas)


def test_registro_cubre_las_cuatro_familias_del_hito() -> None:
    assert set(FABRICAS) == {"classical", "ml", "dl", "foundation"}


def test_linea_base_es_seasonal_naive_de_periodo_siete() -> None:
    serie = _serie(30)
    linea_base = instanciar_linea_base()
    assert isinstance(linea_base, Pronosticador)
    pronostico = linea_base.fit(serie).predict(10)
    ultimo_ciclo = serie[-PERIODO_LINEA_BASE:]
    assert np.array_equal(pronostico, np.tile(ultimo_ciclo, 2)[:10])
