"""Metricas 3.4-A1 (ADR-03-006): exactas, solo sobre la reserva y sin epsilon."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from tests._evaluaciones import (
    T_ESTRELLA,
    corte,
    linea_base,
    pronostico,
    reserva,
    serie,
)

from pred_engine.forecasting.evaluaciones.calculo_errores import (
    EntradaSku,
    EvaluacionRetrospectivaError,
    escala_q1,
    evaluar_sku,
    separar_reserva,
)

# Historia 0..13: |y_t - y_{t-7}| = 7 en todo t, asi Q1 = 7.
_HISTORIA = np.arange(14, dtype=float)


def _entrada(candidatos, base, reales=(10.0, 10.0, 10.0, 10.0), **kwargs):
    kwargs.setdefault("historia", _HISTORIA)
    return EntradaSku(
        sku="S1",
        sku_class="smooth",
        reserva=reserva(reales),
        linea_base=linea_base(base),
        candidatos=tuple(candidatos),
        **kwargs,
    )


def _evaluar(candidatos, base, **kwargs):
    return evaluar_sku(_entrada(candidatos, base, **kwargs), corte())


# --- Lectura segura de la reserva ------------------------------------------


def test_separar_reserva_corta_en_t_estrella() -> None:
    fechas = pd.date_range(T_ESTRELLA - pd.Timedelta(days=2), periods=5, freq="D")
    serie_panel = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0], index=fechas[::-1] + pd.Timedelta(hours=6)
    )
    historia, reservada = separar_reserva(serie_panel, corte())
    assert historia.tolist() == [5.0, 4.0, 3.0]
    assert reservada.tolist() == [2.0, 1.0]
    assert reservada.index.min() > T_ESTRELLA


def test_la_escala_no_cambia_si_cambia_la_reserva() -> None:
    fechas = pd.date_range(T_ESTRELLA - pd.Timedelta(days=13), periods=20, freq="D")
    original = pd.Series(np.arange(20, dtype=float), index=fechas)
    alterada = original.copy()
    alterada.iloc[14:] = 999.0
    assert escala_q1(separar_reserva(original, corte())[0]) == escala_q1(
        separar_reserva(alterada, corte())[0]
    )


# --- Escala Q1 ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("historia", "esperado"),
    [
        (_HISTORIA, (7.0, None)),
        (np.arange(7, dtype=float), (None, "historia_corta")),
        (np.full(20, 3.0), (None, "escala_cero")),
        (np.r_[np.zeros(10), np.inf], (None, "escala_no_finita")),
    ],
)
def test_escala_q1(historia: np.ndarray, esperado: tuple) -> None:
    assert escala_q1(historia) == esperado


# --- Metricas calculadas a mano -------------------------------------------------


def test_metricas_exactas_frente_a_valores_a_mano() -> None:
    evaluado = _evaluar(
        [serie("S1/c", [pronostico(0, [8.0, 9.0, 11.0, 10.0])])],
        [pronostico(0, [6.0, 6.0, 6.0, 6.0])],
    )
    (candidato,) = evaluado.candidatos
    m = candidato.agregadas
    assert m is not None
    # e = [2, 1, -1, 0]
    assert m.mae == pytest.approx(1.0)
    assert m.rmse == pytest.approx(math.sqrt(1.5))
    assert m.me == pytest.approx(0.5)
    assert m.mase == pytest.approx(1.0 / 7.0)
    assert m.razon_sn == pytest.approx(math.sqrt(1.5) / 4.0)
    assert m.n_pares == 4
    assert m.no_calculables == {}


def test_me_positivo_significa_sub_pronostico() -> None:
    evaluado = _evaluar(
        [serie("S1/bajo", [pronostico(0, [5.0] * 4)])],
        [pronostico(0, [6.0] * 4)],
    )
    m = evaluado.candidatos[0].agregadas
    assert m is not None and m.me > 0


def test_linea_base_tiene_razon_uno() -> None:
    evaluado = _evaluar([], [pronostico(0, [6.0] * 4)])
    m = evaluado.linea_base.agregadas
    assert m is not None and m.razon_sn == pytest.approx(1.0)


def test_q1_cero_deja_mase_no_calculable_sin_epsilon() -> None:
    evaluado = _evaluar(
        [serie("S1/c", [pronostico(0, [9.0] * 4)])],
        [pronostico(0, [6.0] * 4)],
        historia=np.full(14, 5.0),
    )
    m = evaluado.candidatos[0].agregadas
    assert m is not None
    assert m.mase is None
    assert m.no_calculables == {"mase": "escala_cero"}
    assert m.razon_sn is not None


def test_linea_base_perfecta_deja_r_no_calculable() -> None:
    evaluado = _evaluar(
        [serie("S1/c", [pronostico(0, [9.0] * 4)])],
        [pronostico(0, [10.0] * 4)],
    )
    m = evaluado.candidatos[0].agregadas
    assert m is not None
    assert m.razon_sn is None
    assert m.no_calculables["razon_sn"] == "rmse_linea_base_cero"


def test_reserva_con_ceros_no_divide_por_cero() -> None:
    evaluado = _evaluar(
        [serie("S1/c", [pronostico(0, [0.0, 1.0, 0.0, 0.0])])],
        [pronostico(0, [2.0, 0.0, 0.0, 0.0])],
        reales=(0.0, 0.0, 3.0, 0.0),
    )
    assert evaluado.reserva_toda_cero is False
    m = evaluado.candidatos[0].agregadas
    assert m is not None and math.isfinite(m.mae) and math.isfinite(m.rmse)


# --- Ventanas -----------------------------------------------------------------


def _ventanas_basicas() -> list:
    return [pronostico(0, [6.0, 6.0]), pronostico(2, [6.0, 6.0])]


def test_pronostico_no_finito_o_de_otra_longitud_se_excluye_y_cuenta() -> None:
    candidato = serie(
        "S1/c",
        [
            pronostico(0, [9.0, 9.0]),
            pronostico(1, [np.nan, 9.0]),
            pronostico(2, [9.0, 9.0]),
        ],
    )
    corto = pronostico(1, [9.0, 9.0])
    otro = serie(
        "S1/d",
        [
            pronostico(0, [9.0, 9.0]),
            type(corto)(corto.origen, corto.fechas, np.array([9.0])),
        ],
    )
    evaluado = _evaluar([candidato, otro], _ventanas_basicas())
    primero, segundo = evaluado.candidatos
    assert (primero.n_ventanas_totales, primero.n_ventanas_validas) == (3, 2)
    assert primero.n_pronosticos_validos == 2
    assert (segundo.n_ventanas_validas, segundo.n_pronosticos_validos) == (1, 1)
    assert primero.cobertura == pytest.approx(2 / 3)


def test_falta_de_observaciones_no_es_falla_del_modelo() -> None:
    # La ventana pide dias despues de la ultima observacion real.
    evaluado = _evaluar(
        [serie("S1/c", [pronostico(3, [9.0, 9.0])])],
        [pronostico(3, [6.0, 6.0])],
    )
    (candidato,) = evaluado.candidatos
    assert candidato.n_pronosticos_validos == 1
    assert candidato.n_ventanas_validas == 0
    assert candidato.agregadas is None


def test_metricas_por_ventana_y_razon_solo_en_ventanas_comunes() -> None:
    evaluado = _evaluar(
        [
            serie(
                "S1/c",
                [pronostico(0, [8.0, 8.0]), pronostico(2, [9.0, 9.0])],
            )
        ],
        # La linea base solo cubre el primer origen.
        [pronostico(0, [6.0, 6.0]), pronostico(2, [np.inf, 6.0])],
    )
    (candidato,) = evaluado.candidatos
    primera, segunda = candidato.por_ventana
    assert primera.metricas.mae == pytest.approx(2.0)
    assert primera.metricas.razon_sn == pytest.approx(0.5)
    assert segunda.metricas.razon_sn is None
    assert segunda.metricas.no_calculables["razon_sn"].startswith("sin_ventanas")
    agregadas = candidato.agregadas
    assert agregadas is not None
    # MAE sobre las dos ventanas; r solo sobre la comun (2 / 4).
    assert agregadas.mae == pytest.approx(1.5)
    assert agregadas.razon_sn == pytest.approx(0.5)


def test_un_solo_origen_en_t_estrella_es_una_ventana() -> None:
    # Lo que entrega hoy el pipeline: un pronostico desde t* para toda la reserva.
    evaluado = _evaluar(
        [serie("S1/c", [pronostico(0, [9.0] * 4)])],
        [pronostico(0, [6.0] * 4)],
    )
    assert evaluado.candidatos[0].n_ventanas_validas == 1


def test_conteo_de_observaciones_y_reserva_toda_cero() -> None:
    evaluado = _evaluar([], [pronostico(0, [1.0] * 4)], reales=(0.0, np.nan, 0.0, 0.0))
    assert evaluado.n_obs_validas_reserva == 3
    assert evaluado.reserva_toda_cero is True


# --- Causalidad y contrato --------------------------------------------------


@pytest.mark.parametrize("dias", [-1, -3])
def test_origen_antes_de_t_estrella_se_rechaza(dias: int) -> None:
    with pytest.raises(EvaluacionRetrospectivaError, match="t\\*"):
        _evaluar(
            [serie("S1/c", [pronostico(dias, [9.0] * 4)])], [pronostico(0, [6.0] * 4)]
        )


def test_fechas_dentro_de_la_historia_se_rechazan() -> None:
    p = pronostico(0, [9.0, 9.0])
    fuera = type(p)(p.origen, p.fechas - pd.Timedelta(days=1), p.valores)
    with pytest.raises(EvaluacionRetrospectivaError):
        _evaluar([serie("S1/c", [fuera])], [pronostico(0, [6.0, 6.0])])


def test_origen_repetido_se_rechaza(monkeypatch: pytest.MonkeyPatch) -> None:
    errores: list[str] = []

    class _Captura:
        def info(self, *args: object) -> None:
            pass

        def error(self, mensaje: str, *args: object) -> None:
            errores.append(mensaje % args)

    monkeypatch.setattr(
        "pred_engine.forecasting.evaluaciones.calculo_errores.metricas._logger",
        _Captura(),
    )
    repetido = [pronostico(0, [9.0] * 2), pronostico(0, [8.0] * 2)]
    with pytest.raises(EvaluacionRetrospectivaError, match="origen repetido"):
        _evaluar([serie("S1/c", repetido)], [pronostico(0, [6.0] * 2)])
    assert len(errores) == 1 and "candidato_id=S1/c" in errores[0]


def test_candidatos_fallidos_se_conservan_en_la_evaluacion() -> None:
    evaluado = _evaluar([], [pronostico(0, [6.0] * 4)], n_candidatos_fallidos=2)
    assert evaluado.n_candidatos_fallidos == 2
