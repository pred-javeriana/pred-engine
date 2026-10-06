"""Veredictos 3.4-A2 (ADR-03-008): reglas en orden, evidencia visible por SKU."""

from __future__ import annotations

import numpy as np
import pytest
from tests._evaluaciones import (
    corte,
    evaluacion,
    linea_base,
    metricas_candidato,
    pronostico,
    reserva,
    serie,
)

from pred_engine.forecasting.evaluaciones.calculo_errores import (
    FAMILIA_LINEA_BASE,
    VERSION_METRICAS,
    EntradaSku,
    EvaluacionRetrospectivaError,
    evaluar_sku,
)
from pred_engine.forecasting.evaluaciones.veredictos import (
    POLITICA_INICIAL,
    VEREDICTOS,
    emitir_veredictos,
)


def _unico(*evaluaciones, datos_sinteticos: bool = False):
    resultado = emitir_veredictos(evaluaciones, datos_sinteticos=datos_sinteticos)
    return {v.sku: v for v in resultado.skus}


def test_validado_con_datos_reales_suficientes_y_r_menor_que_uno() -> None:
    veredicto = _unico(evaluacion("A", [metricas_candidato("ml", 0.8)]))["A"]
    assert veredicto.veredicto == "VALIDADO"
    assert veredicto.familia_campeona == "ml"
    assert veredicto.candidato_campeon == "S/ml"
    assert veredicto.razon_sn == pytest.approx(0.8)
    assert veredicto.pierde_frente_a_linea_base is False
    assert veredicto.n_ventanas == 40
    assert veredicto.cobertura == pytest.approx(1.0)
    assert veredicto.metricas_campeon is not None
    assert veredicto.metricas_linea_base is not None


def test_datos_sinteticos_nunca_pasan_de_exploratorio() -> None:
    veredicto = _unico(
        evaluacion("A", [metricas_candidato("ml", 0.8)]), datos_sinteticos=True
    )["A"]
    assert veredicto.veredicto == "EXPLORATORIO"
    assert "sinteticos" in veredicto.justificacion


def test_comparacion_incompleta_limita_a_exploratorio() -> None:
    veredicto = _unico(evaluacion("A", [metricas_candidato("ml", 0.8)], fallidos=1))[
        "A"
    ]
    assert veredicto.comparacion_incompleta is True
    assert veredicto.veredicto == "EXPLORATORIO"


def test_candidato_sin_pronostico_valido_marca_comparacion_incompleta() -> None:
    veredicto = _unico(
        evaluacion(
            "A",
            [
                metricas_candidato("ml", 0.8),
                metricas_candidato("dl", None, n=0, pronosticos_validos=0),
            ],
        )
    )["A"]
    assert veredicto.comparacion_incompleta is True


def test_categoria_adversa_deja_al_sku_exploratorio_con_la_linea_base() -> None:
    veredicto = _unico(evaluacion("A", [metricas_candidato("ml", 1.3)]))["A"]
    assert veredicto.familia_campeona == FAMILIA_LINEA_BASE
    assert veredicto.veredicto == "EXPLORATORIO"
    assert veredicto.pierde_frente_a_linea_base is True


def test_pocas_ventanas_es_evidencia_insuficiente() -> None:
    # Hoy el pipeline entrega una sola ventana por candidato (origen t*).
    veredicto = _unico(evaluacion("A", [metricas_candidato("ml", 0.5, n=1)]))["A"]
    assert veredicto.veredicto == "EVIDENCIA_INSUFICIENTE"
    assert f"n_min={POLITICA_INICIAL.n_min}" in veredicto.justificacion


def test_reserva_toda_en_cero_es_evidencia_insuficiente() -> None:
    veredicto = _unico(
        evaluacion("A", [metricas_candidato("ml", 0.5)], toda_cero=True)
    )["A"]
    assert veredicto.veredicto == "EVIDENCIA_INSUFICIENTE"


def test_sin_observaciones_validas_es_no_evaluable() -> None:
    veredicto = _unico(evaluacion("A", [metricas_candidato("ml", 0.5)], n_obs=0))["A"]
    assert veredicto.veredicto == "NO_EVALUABLE"


def test_todos_los_candidatos_fallaron_es_fallo_tecnico_aunque_haya_datos() -> None:
    sin_pronostico = metricas_candidato("ml", None, n=0, pronosticos_validos=0)
    assert _unico(evaluacion("A", [sin_pronostico]))["A"].veredicto == "FALLO_TECNICO"
    assert _unico(evaluacion("B", [], fallidos=2))["B"].veredicto == "FALLO_TECNICO"


def test_fallo_tecnico_precede_a_no_evaluable() -> None:
    veredicto = _unico(evaluacion("A", [], fallidos=1, n_obs=0))["A"]
    assert veredicto.veredicto == "FALLO_TECNICO"


def test_sku_sin_instancia_de_la_familia_campeona_es_fallo_tecnico() -> None:
    veredictos = _unico(
        evaluacion("A", [metricas_candidato("ml", 0.5), metricas_candidato("dl", 0.9)]),
        evaluacion("B", [metricas_candidato("dl", 0.9)]),
    )
    assert veredictos["A"].familia_campeona == "ml"
    assert veredictos["B"].veredicto == "FALLO_TECNICO"
    assert veredictos["B"].candidato_campeon is None


def test_cada_sku_lista_todos_sus_modelos_evaluados_con_su_r() -> None:
    # B tiene su mejor r en SARIMA, aunque la categoria la gane Chronos.
    veredictos = _unico(
        evaluacion(
            "A",
            [
                metricas_candidato("classical", 0.9),
                metricas_candidato("foundation", 0.6),
            ],
        ),
        evaluacion(
            "B",
            [
                metricas_candidato("classical", 0.5),
                metricas_candidato("foundation", 0.7),
                # Una segunda configuracion que no pronostico: r no calculable.
                metricas_candidato(
                    "classical",
                    None,
                    n=0,
                    pronosticos_validos=0,
                    candidato_id="S/classical/2",
                ),
            ],
        ),
    )
    b = veredictos["B"]
    assert b.familia_campeona == "foundation"
    assert [(m.candidato_id, m.razon_sn) for m in b.modelos_evaluados] == [
        ("S/seasonal_naive", 1.0),
        ("S/classical", 0.5),
        ("S/foundation", 0.7),
        ("S/classical/2", None),
    ]
    assert [m.candidato_id for m in veredictos["A"].modelos_evaluados] == [
        "S/seasonal_naive",
        "S/classical",
        "S/foundation",
    ]


def test_iqr_de_la_diferencia_de_mae_por_ventana() -> None:
    campeon = metricas_candidato("ml", 0.5, n=4, maes=[1.0, 2.0, 3.0, 4.0])
    base = metricas_candidato(FAMILIA_LINEA_BASE, 1.0, n=4, maes=[1.0] * 4)
    veredicto = _unico(evaluacion("A", [campeon], base=base))["A"]
    # Diferencias [0, 1, 2, 3]: Q3 - Q1 = 2.25 - 0.75.
    assert veredicto.iqr_diferencia_mae == pytest.approx(1.5)


def test_resumen_por_categoria() -> None:
    resultado = emitir_veredictos(
        [
            evaluacion("A", [metricas_candidato("ml", 0.5)]),
            evaluacion("B", [metricas_candidato("ml", 1.2)]),
            evaluacion("C", [metricas_candidato("ml", 0.7)], n_obs=0),
            evaluacion("D", [metricas_candidato("ml", 0.9)], sku_class="smooth"),
        ],
        datos_sinteticos=False,
    )
    assert [c.sku_class for c in resultado.categorias] == ["lumpy", "smooth"]
    lumpy = resultado.categorias[0]
    assert lumpy.seleccion.familia_campeona == "ml"
    assert lumpy.conteos == {
        "FALLO_TECNICO": 0,
        "NO_EVALUABLE": 1,
        "EVIDENCIA_INSUFICIENTE": 0,
        "VALIDADO": 1,
        "EXPLORATORIO": 1,
    }
    assert sum(lumpy.porcentajes.values()) == pytest.approx(100.0)
    assert lumpy.mediana_r == pytest.approx(0.7)
    assert lumpy.n_adversos == 1
    assert set(lumpy.conteos) == set(VEREDICTOS)
    assert resultado.version_politica == POLITICA_INICIAL.version
    assert resultado.version_metricas == VERSION_METRICAS


def test_sku_repetido_se_rechaza() -> None:
    with pytest.raises(EvaluacionRetrospectivaError):
        emitir_veredictos(
            [evaluacion("A", []), evaluacion("A", [])], datos_sinteticos=False
        )


def test_flujo_de_metricas_a_veredicto_sobre_35_ventanas() -> None:
    # Integracion 3.4-A1 -> A2: 35 ventanas de un dia, candidato mejor que SN.
    reales = 10 + np.sin(np.arange(36))
    entrada = EntradaSku(
        sku="A",
        sku_class="smooth",
        historia=np.arange(30, dtype=float) % 7,
        reserva=reserva(reales),
        linea_base=linea_base([pronostico(d, [reales[d] + 2.0]) for d in range(35)]),
        candidatos=(
            serie(
                "A/ml/lightgbm",
                [pronostico(d, [reales[d] + 0.5]) for d in range(35)],
                familia="ml",
                modelo="lightgbm",
            ),
        ),
    )
    evaluado = evaluar_sku(entrada, corte())
    (veredicto,) = emitir_veredictos([evaluado], datos_sinteticos=False).skus
    assert veredicto.veredicto == "VALIDADO"
    assert veredicto.n_ventanas == 35
    assert veredicto.razon_sn == pytest.approx(0.25)
