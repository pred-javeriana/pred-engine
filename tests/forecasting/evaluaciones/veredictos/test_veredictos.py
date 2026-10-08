"""Veredictos 3.4-A2 (ADR-03-008): reglas en orden, evidencia visible por SKU."""

from __future__ import annotations

import numpy as np
import pytest
from tests._evaluaciones import (
    corte,
    evaluacion,
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
from pred_engine.forecasting.evaluaciones.diebold_mariano import prueba_hln
from pred_engine.forecasting.evaluaciones.veredictos import (
    POLITICA_INICIAL,
    VEREDICTOS,
    emitir_veredictos,
)

# RMSE por ventana de un campeon claramente mejor que SN (RMSE 1 por ventana).
_MEJOR = [0.5, 0.7, 0.4, 0.9] * 10


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


def test_la_falla_de_una_familia_no_elegible_no_marca_comparacion_incompleta() -> None:
    # En lumpy la politica no admite DL: su falla no deja incompleta la comparacion.
    veredicto = _unico(
        evaluacion(
            "A",
            [
                metricas_candidato("classical", 0.8),
                metricas_candidato("dl", None, n=0, pronosticos_validos=0),
            ],
            sku_class="lumpy",
        )
    )["A"]
    assert veredicto.comparacion_incompleta is False
    assert veredicto.veredicto == "VALIDADO"


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


def test_diebold_mariano_acompana_al_veredicto_sin_cambiarlo() -> None:
    veredicto = _unico(evaluacion("A", [metricas_candidato("ml", 0.8, rmses=_MEJOR)]))[
        "A"
    ]
    prueba = veredicto.diebold_mariano
    esperado = prueba_hln(np.square(_MEJOR) - 1.0, 1)
    assert veredicto.veredicto == "VALIDADO"
    assert prueba.estadistico == pytest.approx(esperado.estadistico)
    # Una sola prueba en la corrida: BH no cambia el p-valor.
    assert prueba.p_ajustado == pytest.approx(esperado.p_valor)
    assert prueba.significativa is True


def test_sin_diebold_mariano_queda_la_causa() -> None:
    campeon = metricas_candidato("ml", 0.8, rmses=_MEJOR)
    sintetico = _unico(evaluacion("A", [campeon]), datos_sinteticos=True)["A"]
    adverso = _unico(evaluacion("A", [metricas_candidato("ml", 1.3)]))["A"]
    corto = _unico(evaluacion("A", [metricas_candidato("ml", 0.5, n=1)]))["A"]
    assert sintetico.diebold_mariano.causa == "datos_sinteticos"
    assert adverso.diebold_mariano.causa == "campeon_es_linea_base"
    assert corto.diebold_mariano.causa == "veredicto:EVIDENCIA_INSUFICIENTE"
    assert all(
        v.diebold_mariano.p_ajustado is None for v in (sintetico, adverso, corto)
    )


def test_bh_corrige_entre_todos_los_skus_de_la_corrida() -> None:
    ruido = [1.2, 0.7, 1.1, 0.8, 0.95] * 8
    veredictos = _unico(
        evaluacion("A", [metricas_candidato("ml", 0.8, rmses=_MEJOR)]),
        evaluacion(
            "B", [metricas_candidato("ml", 0.9, rmses=ruido)], sku_class="erratic"
        ),
    )
    p_a = prueba_hln(np.square(_MEJOR) - 1.0, 1).p_valor
    p_b = prueba_hln(np.square(ruido) - 1.0, 1).p_valor
    assert p_a is not None and p_b is not None and p_a < p_b
    # BH con m=2, aunque esten en categorias distintas.
    assert veredictos["A"].diebold_mariano.p_ajustado == pytest.approx(
        min(2 * p_a, p_b)
    )
    assert veredictos["B"].diebold_mariano.p_ajustado == pytest.approx(p_b)


def test_resumen_por_categoria() -> None:
    resultado = emitir_veredictos(
        [
            evaluacion("A", [metricas_candidato("ml", 0.5)]),
            evaluacion("B", [metricas_candidato("ml", 1.2)]),
            evaluacion("C", [metricas_candidato("ml", 0.7)], n_obs=0),
            evaluacion("D", [metricas_candidato("ml", 0.9)], sku_class="erratic"),
        ],
        datos_sinteticos=False,
    )
    assert [c.sku_class for c in resultado.categorias] == ["erratic", "smooth"]
    smooth = resultado.categorias[1]
    assert smooth.seleccion.familia_campeona == "ml"
    assert smooth.conteos == {
        "FALLO_TECNICO": 0,
        "NO_EVALUABLE": 1,
        "EVIDENCIA_INSUFICIENTE": 0,
        "VALIDADO": 1,
        "EXPLORATORIO": 1,
    }
    assert sum(smooth.porcentajes.values()) == pytest.approx(100.0)
    assert smooth.mediana_r == pytest.approx(0.7)
    assert smooth.n_adversos == 1
    assert set(smooth.conteos) == set(VEREDICTOS)
    assert resultado.version_politica == POLITICA_INICIAL.version
    assert resultado.version_metricas == VERSION_METRICAS
    assert resultado.datos_sinteticos is False


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
        historia=np.full(30, 10.0),
        reserva=reserva(reales),
        candidatos=(
            serie(
                "A/ml/lightgbm",
                [pronostico(d, [reales[d] + 0.1]) for d in range(35)],
                familia="ml",
                modelo="lightgbm",
            ),
        ),
    )
    evaluado = evaluar_sku(entrada, corte())
    (veredicto,) = emitir_veredictos([evaluado], datos_sinteticos=False).skus
    # Seasonal Naive (m=7): 10 desde la historia, luego lo observado 7 dias antes.
    linea_base = np.array([10.0 if d < 7 else reales[d - 7] for d in range(35)])
    rmse_sn = np.sqrt(np.mean((reales[:35] - linea_base) ** 2))
    assert veredicto.veredicto == "VALIDADO"
    assert veredicto.n_ventanas == 35
    assert veredicto.razon_sn == pytest.approx(0.1 / rmse_sn)
    assert veredicto.diebold_mariano.n_ventanas == 35
    assert veredicto.diebold_mariano.horizonte == 1
    assert veredicto.diebold_mariano.significativa is True
