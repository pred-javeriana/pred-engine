"""Vista de los pliegues L3 previos a t* para la seleccion de robustez (ADR-023)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests._evaluaciones import T_ESTRELLA

from pred_engine.comun.dataclasses.validacion_temporal import (
    ResultadoVentana,
    ResultadoWalkForward,
    VentanaTemporal,
)
from pred_engine.forecasting.evaluaciones.veredictos import (
    POLITICA_INICIAL,
    CandidatoPrevio,
    evaluacion_previa,
    seleccionar_categoria,
    seleccionar_previa,
)

# 100 dias hasta t*, con semana marcada: Seasonal Naive no es trivial. Los 42
# pliegues superan el minimo de 30 ventanas comunes de la seleccion.
_HISTORIA = 10.0 + (np.arange(100) % 7) + np.sin(np.arange(100))
_CORTES = tuple(range(14, 98, 2))


def _walk_forward(sesgo: float, fallidos: tuple[int, ...] = ()) -> ResultadoWalkForward:
    ventanas = []
    for i, fin in enumerate(_CORTES):
        y_real = _HISTORIA[fin : fin + 2]
        ventanas.append(
            ResultadoVentana(
                ventana=VentanaTemporal(i, 0, fin, fin, fin + 2),
                metricas={},
                y_train=_HISTORIA[:fin],
                y_real=y_real,
                y_pred=y_real + sesgo,
                duracion_s=0.0,
                fallo="ajuste" if fin in fallidos else None,
            )
        )
    return ResultadoWalkForward(tuple(ventanas), "mae", 0.0, True, 42, 42)


def _previa(sku: str, sesgos: dict[str, float], fallidos: tuple[int, ...] = ()):
    return evaluacion_previa(
        sku,
        "smooth",
        T_ESTRELLA,
        _HISTORIA,
        [
            CandidatoPrevio(f"{sku}/{f}/{f}", f, f, _walk_forward(s, fallidos))
            for f, s in sesgos.items()
        ],
    )


def test_cada_pliegue_es_una_ventana_con_su_linea_base() -> None:
    previa = _previa("A", {"ml": 0.5, "dl": 1.0}, fallidos=(20,))
    ml, dl = previa.candidatos
    # El origen de un pliegue es la fecha de su ultima observacion de entrenamiento.
    assert [v.origen for v in ml.por_ventana][:2] == [
        T_ESTRELLA - pd.Timedelta(days=100 - 14),
        T_ESTRELLA - pd.Timedelta(days=100 - 16),
    ]
    assert previa.linea_base.n_ventanas_validas == len(_CORTES)
    # Seasonal Naive ajustado en cada pliegue: repite la semana previa al origen.
    errores_sn = [
        np.abs(_HISTORIA[f : f + 2] - _HISTORIA[f - 7 : f - 5]).mean() for f in _CORTES
    ]
    assert [v.metricas.mae for v in previa.linea_base.por_ventana] == pytest.approx(
        errores_sn
    )
    # Un pliegue fallido no cuenta para su candidato.
    assert (dl.n_ventanas_totales, dl.n_pronosticos_validos) == (42, 41)
    assert dl.agregadas is not None and dl.agregadas.mae == pytest.approx(1.0)
    assert ml.agregadas is not None and ml.agregadas.razon_sn is not None
    assert dl.agregadas.razon_sn is not None
    assert ml.agregadas.razon_sn < dl.agregadas.razon_sn
    assert (previa.n_obs_validas_reserva, previa.reserva_toda_cero) == (84, False)


def test_la_regla_de_seleccion_lee_la_vista_previa() -> None:
    previas = [_previa(s, {"ml": 1.0, "dl": 0.3}) for s in ("A", "B", "C")]
    seleccion = seleccionar_categoria("smooth", previas, POLITICA_INICIAL)
    assert seleccion.familia_campeona == "dl"
    assert seleccion.skus_comparables == ("A", "B", "C")


def test_cada_categoria_recibe_su_seleccion_previa() -> None:
    previas = [_previa(s, {"ml": 1.0, "dl": 0.3}) for s in ("A", "B", "C")]
    lumpy, smooth = seleccionar_previa(previas, ["smooth", "lumpy", "smooth"])
    assert smooth == seleccionar_categoria("smooth", previas, POLITICA_INICIAL)
    # Sin pliegues de la categoria no hay SKU comparables.
    assert (lumpy.sku_class, lumpy.motivo, lumpy.skus_comparables) == (
        "lumpy",
        "sin_skus_comparables",
        (),
    )
