"""Bloques de seleccion y de prueba de la reserva (ADR-023)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests._evaluaciones import T_ESTRELLA, corte, pronostico, reserva, serie

from pred_engine.forecasting.evaluaciones.calculo_errores import (
    BloquesReserva,
    EntradaSku,
    EvaluacionRetrospectivaError,
    evaluar_bloques,
)


def test_limites_de_la_reserva_de_338_dias_con_horizonte_7() -> None:
    bloques = BloquesReserva(pd.Timestamp("2016-09-11"), 338, 7, 0.4)
    assert bloques.dias_seleccion == 135
    assert bloques.limites() == {
        "fraccion_seleccion": 0.4,
        "dias_reserva": 338,
        "horizonte": 7,
        "dias_seleccion": 135,
        "origenes_seleccion": {
            "n": 129,
            "primero": "2016-09-11",
            "ultimo": "2017-01-17",
        },
        "origenes_embargo": {"n": 6, "primero": "2017-01-18", "ultimo": "2017-01-23"},
        "origenes_prueba": {"n": 197, "primero": "2017-01-24", "ultimo": "2017-08-08"},
        "fechas_seleccion": {"n": 135, "primero": "2016-09-12", "ultimo": "2017-01-24"},
        "fechas_prueba": {"n": 203, "primero": "2017-01-25", "ultimo": "2017-08-15"},
    }


@pytest.mark.parametrize(
    ("k", "bloque"),
    [
        (0, "seleccion"),
        (128, "seleccion"),
        (129, "embargo"),
        (134, "embargo"),
        (135, "prueba"),
        (331, "prueba"),
    ],
)
def test_cada_origen_pertenece_a_un_bloque(k: int, bloque: str) -> None:
    t_star = pd.Timestamp("2016-09-11")
    bloques = BloquesReserva(t_star, 338, 7, 0.4)
    assert bloques.bloque(t_star + pd.Timedelta(days=k)) == bloque


@pytest.mark.parametrize(
    "argumentos",
    [
        {"dias_reserva": 0},
        {"dias_reserva": True},
        {"horizonte": 0},
        {"fraccion_seleccion": 0.0},
        {"fraccion_seleccion": 1.0},
    ],
)
def test_bloques_invalidos_se_rechazan(argumentos: dict) -> None:
    datos = {"dias_reserva": 10, "horizonte": 2, "fraccion_seleccion": 0.4}
    with pytest.raises(ValueError):
        BloquesReserva(T_ESTRELLA, **(datos | argumentos))


def _entrada() -> EntradaSku:
    # Reserva de 10 dias: 3 en los dias de seleccion y 0 en los de prueba.
    reales = [3.0] * 4 + [0.0] * 6
    return EntradaSku(
        sku="S1",
        sku_class="smooth",
        historia=np.full(14, 3.0),
        reserva=reserva(reales),
        candidatos=(
            serie(
                "S1/ml/lightgbm",
                [pronostico(k, [4.0, 4.0]) for k in range(9)],
                familia="ml",
                modelo="lightgbm",
            ),
        ),
    )


def test_cada_bloque_solo_lee_sus_ventanas_y_sus_dias() -> None:
    # R = 10, H = 2, f = 0.4: S = 4. Seleccion k = 0..2, embargo k = 3, prueba 4..8.
    bloques = BloquesReserva(T_ESTRELLA, 10, 2, 0.4)
    evaluado = evaluar_bloques(_entrada(), corte(10), bloques)

    def origenes(vista) -> list[int]:
        (candidato,) = vista.candidatos
        return [(v.origen - T_ESTRELLA).days for v in candidato.por_ventana]

    assert origenes(evaluado.reserva) == list(range(9))
    assert origenes(evaluado.seleccion) == [0, 1, 2]
    assert origenes(evaluado.prueba) == [4, 5, 6, 7, 8]
    seleccion, prueba = evaluado.seleccion, evaluado.prueba
    assert (seleccion.n_obs_validas_reserva, seleccion.reserva_toda_cero) == (4, False)
    assert (prueba.n_obs_validas_reserva, prueba.reserva_toda_cero) == (6, True)
    assert seleccion.candidatos[0].n_ventanas_totales == 3
    assert prueba.candidatos[0].n_ventanas_totales == 5
    # El error 1 de la seleccion y el 4 de la prueba no se mezclan.
    assert seleccion.candidatos[0].agregadas is not None
    assert prueba.candidatos[0].agregadas is not None
    assert seleccion.candidatos[0].agregadas.mae == pytest.approx(1.0)
    assert prueba.candidatos[0].agregadas.mae == pytest.approx(4.0)
    assert evaluado.sku == "S1" and evaluado.sku_class == "smooth"


def test_bloques_de_otra_reserva_se_rechazan() -> None:
    with pytest.raises(EvaluacionRetrospectivaError, match="otra reserva"):
        evaluar_bloques(_entrada(), corte(10), BloquesReserva(T_ESTRELLA, 12, 2, 0.4))
