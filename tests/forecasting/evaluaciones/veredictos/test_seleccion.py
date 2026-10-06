"""Seleccion por categoria 3.4-A2 (ADR-03-007): mediana de r con compuerta."""

from __future__ import annotations

import pytest
from tests._evaluaciones import evaluacion, metricas_candidato

from pred_engine.forecasting.evaluaciones.calculo_errores import FAMILIA_LINEA_BASE
from pred_engine.forecasting.evaluaciones.veredictos import (
    POLITICA_INICIAL,
    PoliticaSeleccion,
    representante,
    seleccionar_categoria,
)


def _sku(sku: str, **razones: float | None):
    return evaluacion(sku, [metricas_candidato(f, r) for f, r in razones.items()])


def _seleccionar(*evaluaciones, politica: PoliticaSeleccion = POLITICA_INICIAL):
    return seleccionar_categoria("lumpy", evaluaciones, politica)


def test_ejemplo_del_adr_gana_chronos_con_mediana_085() -> None:
    seleccion = _seleccionar(
        _sku("A", classical=0.80, foundation=0.90),
        _sku("B", classical=1.10, foundation=0.70),
        _sku("C", classical=0.95, foundation=0.85),
    )
    assert seleccion.familia_campeona == "foundation"
    assert seleccion.medianas_r == pytest.approx(
        {"classical": 0.95, "foundation": 0.85}
    )
    assert seleccion.motivo == "menor_mediana"
    assert seleccion.adverso is False
    assert seleccion.skus_comparables == ("A", "B", "C")


def test_representante_es_la_configuracion_de_menor_rmse() -> None:
    dos_configuraciones = evaluacion(
        "A",
        [
            metricas_candidato("ml", 0.5, rmse=2.0, candidato_id="A/ml/1"),
            metricas_candidato("ml", 0.9, rmse=1.0, candidato_id="A/ml/2"),
        ],
    )
    elegido = representante(dos_configuraciones, "ml")
    assert elegido is not None and elegido.candidato_id == "A/ml/2"
    assert representante(dos_configuraciones, "dl") is None


def test_sku_sin_r_calculable_sale_del_conjunto_comun_con_causa() -> None:
    seleccion = _seleccionar(
        _sku("A", classical=0.8, ml=0.6),
        _sku("B", classical=0.9, ml=None),
        evaluacion("C", [metricas_candidato("classical", 0.7)]),
    )
    assert seleccion.skus_comparables == ("A",)
    assert seleccion.excluidos == {
        "B": "sin_razon_calculable:ml",
        "C": "sin_razon_calculable:ml",
    }
    assert seleccion.familia_campeona == "ml"


def test_compuerta_de_linea_base_si_ninguna_familia_mejora() -> None:
    seleccion = _seleccionar(
        _sku("A", classical=1.2, ml=1.0), _sku("B", classical=1.1, ml=1.3)
    )
    assert seleccion.familia_campeona == FAMILIA_LINEA_BASE
    assert seleccion.adverso is True
    assert seleccion.motivo == "compuerta_linea_base"


def test_empate_dentro_de_la_tolerancia_gana_la_familia_mas_simple() -> None:
    seleccion = _seleccionar(_sku("A", classical=0.800, ml=0.795))
    assert seleccion.familia_campeona == "classical"
    assert seleccion.motivo == "empate_por_simplicidad"


def test_el_desempate_no_elige_una_familia_peor_que_la_linea_base() -> None:
    seleccion = _seleccionar(_sku("A", classical=1.004, ml=0.995))
    assert seleccion.familia_campeona == "ml"
    assert seleccion.motivo == "menor_mediana"


def test_fuera_de_la_tolerancia_gana_la_menor_mediana() -> None:
    seleccion = _seleccionar(_sku("A", classical=0.80, ml=0.70))
    assert seleccion.familia_campeona == "ml"


def test_sin_skus_comparables_el_campeon_es_la_linea_base() -> None:
    seleccion = _seleccionar(_sku("A", classical=None), evaluacion("B", []))
    assert seleccion.familia_campeona == FAMILIA_LINEA_BASE
    assert seleccion.motivo == "sin_skus_comparables"
    assert seleccion.medianas_r == {}


def test_familia_fuera_del_orden_declarado_es_la_menos_simple() -> None:
    politica = PoliticaSeleccion(orden_simplicidad=("ml",))
    seleccion = _seleccionar(_sku("A", classical=0.800, ml=0.805), politica=politica)
    assert seleccion.familia_campeona == "ml"


@pytest.mark.parametrize(
    "argumentos",
    [
        {"version": " "},
        {"n_min": 0},
        {"tolerancia_empate": -0.1},
        {"tolerancia_empate": float("nan")},
        {"orden_simplicidad": ("ml", "ml")},
    ],
)
def test_politica_invalida_se_rechaza(argumentos: dict) -> None:
    with pytest.raises(ValueError):
        PoliticaSeleccion(**argumentos)


def test_politica_inicial_predeclarada() -> None:
    assert POLITICA_INICIAL.n_min == 30
    assert POLITICA_INICIAL.orden_simplicidad[0] == "classical"
