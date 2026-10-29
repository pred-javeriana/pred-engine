"""Veredictos 3.4-A2 (ADR-03-008, ADR-023): campeon congelado y bloque de prueba."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests
from tests._evaluaciones import (
    BLOQUES,
    corte,
    en_bloques,
    evaluacion,
    metricas_candidato,
    pronostico,
    reserva,
    serie,
)

from pred_engine.forecasting.evaluaciones.calculo_errores import (
    FAMILIA_LINEA_BASE,
    VERSION_METRICAS,
    BloquesReserva,
    EntradaSku,
    EvaluacionRetrospectivaError,
    evaluar_bloques,
)
from pred_engine.forecasting.evaluaciones.diebold_mariano import prueba_hln
from pred_engine.forecasting.evaluaciones.veredictos import (
    POLITICA_INICIAL,
    VEREDICTOS,
    PoliticaSeleccion,
    emitir_veredictos,
    seleccionar_previa,
)

# RMSE por ventana de un campeon claramente mejor que SN (RMSE 1 por ventana).
_MEJOR = [0.5, 0.7, 0.4, 0.9] * 10


def _emitir(*evaluaciones, datos_sinteticos: bool = False, **opciones):
    return emitir_veredictos(
        [en_bloques(e) for e in evaluaciones],
        BLOQUES,
        datos_sinteticos=datos_sinteticos,
        **opciones,
    )


def _unico(*evaluaciones, datos_sinteticos: bool = False):
    resultado = _emitir(*evaluaciones, datos_sinteticos=datos_sinteticos)
    return {v.sku: v for v in resultado.skus}


def _ruido(delta: float, fase: int) -> list[float]:
    """RMSE por ventana con mejora media `delta` sobre SN y ruido senoidal."""
    t = np.arange(40)
    return list(np.sqrt(np.clip(1 - delta + 0.25 * np.sin(1.3 * t + fase), 0.05, None)))


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


def test_pocas_ventanas_de_prueba_es_evidencia_insuficiente() -> None:
    # La seleccion tiene 40 ventanas; el bloque de prueba, una sola.
    resultado = emitir_veredictos(
        [
            en_bloques(
                evaluacion("A", [metricas_candidato("ml", 0.5, n=1)]),
                seleccion=evaluacion("A", [metricas_candidato("ml", 0.5)]),
            )
        ],
        BLOQUES,
        datos_sinteticos=False,
    )
    (veredicto,) = resultado.skus
    assert resultado.categorias[0].seleccion.familia_campeona == "ml"
    assert veredicto.veredicto == "EVIDENCIA_INSUFICIENTE"
    assert veredicto.justificacion == (
        f"EVIDENCIA_INSUFICIENTE: 1 ventanas validas de prueba "
        f"(< n_min={POLITICA_INICIAL.n_min})"
    )


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


def test_sku_sin_instancia_usa_la_siguiente_familia_de_la_clase() -> None:
    veredictos = _unico(
        evaluacion("A", [metricas_candidato("ml", 0.5), metricas_candidato("dl", 0.9)]),
        evaluacion("B", [metricas_candidato("dl", 0.9)]),
    )
    assert veredictos["A"].familia_campeona == "ml"
    assert veredictos["B"].veredicto == "VALIDADO"
    assert veredictos["B"].familia_campeona == "dl"
    assert veredictos["B"].candidato_campeon == "S/dl"
    assert veredictos["B"].justificacion == (
        "VALIDADO: datos reales, ventanas suficientes y r < 1 (descriptivo); "
        "fallback de familia ml a dl: sin instancia de la campeona de clase"
    )
    assert "fallback" not in veredictos["A"].justificacion


@pytest.mark.parametrize("segunda_disponible", [True, False])
def test_fallback_sigue_mediana_de_clase_nunca_el_mejor_del_producto(
    segunda_disponible: bool,
) -> None:
    candidatos = [metricas_candidato("foundation", 0.1)]
    if segunda_disponible:
        candidatos.append(metricas_candidato("ml", 1.1))
    resultado = _emitir(
        evaluacion(
            "A",
            [
                metricas_candidato("classical", 0.5),
                metricas_candidato("ml", 0.6),
                metricas_candidato("foundation", 0.9),
            ],
            sku_class="intermittent",
        ),
        evaluacion("B", candidatos, sku_class="intermittent", fallidos=1),
    )
    assert resultado.categorias[0].seleccion.familia_campeona == "classical"
    b = next(v for v in resultado.skus if v.sku == "B")
    destino = "ml" if segunda_disponible else "foundation"
    assert b.familia_campeona == destino
    assert b.candidato_campeon == f"S/{destino}"
    assert b.razon_sn == pytest.approx(1.1 if segunda_disponible else 0.1)
    assert b.veredicto == "EXPLORATORIO"
    assert b.comparacion_incompleta is True
    assert (
        f"fallback de familia classical a {destino}: "
        "sin instancia de la campeona de clase"
    ) in b.justificacion


def test_sin_familia_disponible_en_el_ranking_conserva_fallo_tecnico() -> None:
    veredictos = _unico(
        evaluacion("A", [metricas_candidato("classical", 0.5)], sku_class="lumpy"),
        evaluacion("B", [metricas_candidato("dl", 0.1)], sku_class="lumpy"),
    )
    assert veredictos["B"].veredicto == "FALLO_TECNICO"
    assert veredictos["B"].familia_campeona == "classical"
    assert veredictos["B"].candidato_campeon is None
    assert "fallback" not in veredictos["B"].justificacion


def test_instancia_campeona_fallida_no_dispara_fallback() -> None:
    veredictos = _unico(
        evaluacion("A", [metricas_candidato("ml", 0.5), metricas_candidato("dl", 0.9)]),
        evaluacion(
            "B",
            [
                metricas_candidato("ml", None, pronosticos_validos=0),
                metricas_candidato("dl", 0.1),
            ],
        ),
    )
    assert veredictos["B"].veredicto == "FALLO_TECNICO"
    assert veredictos["B"].familia_campeona == "ml"
    assert veredictos["B"].candidato_campeon == "S/ml"
    assert "fallback" not in veredictos["B"].justificacion


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


def test_la_familia_campeona_se_elige_en_el_bloque_de_seleccion() -> None:
    # En la seleccion gana ML; en la prueba DL seria mejor, pero no se reelige.
    resultado = emitir_veredictos(
        [
            en_bloques(
                evaluacion(
                    "A", [metricas_candidato("ml", 0.9), metricas_candidato("dl", 0.5)]
                ),
                seleccion=evaluacion(
                    "A", [metricas_candidato("ml", 0.6), metricas_candidato("dl", 0.8)]
                ),
            )
        ],
        BLOQUES,
        datos_sinteticos=False,
    )
    assert resultado.categorias[0].seleccion.medianas_r == {"dl": 0.8, "ml": 0.6}
    (veredicto,) = resultado.skus
    assert veredicto.familia_campeona == "ml"
    assert veredicto.razon_sn == pytest.approx(0.9)
    assert resultado.bloques == BLOQUES


def test_la_instancia_de_cada_familia_se_elige_en_el_bloque_de_seleccion() -> None:
    # Dos configuraciones ML: la de menor RMSE de seleccion decide, aunque la
    # otra tenga menor RMSE en el bloque de prueba.
    def dos(rmse_1: float, rmse_2: float):
        return evaluacion(
            "A",
            [
                metricas_candidato("ml", 0.7, rmse=rmse_1, candidato_id="A/ml/1"),
                metricas_candidato("ml", 0.8, rmse=rmse_2, candidato_id="A/ml/2"),
            ],
        )

    resultado = emitir_veredictos(
        [en_bloques(dos(2.0, 1.0), seleccion=dos(1.0, 2.0))],
        BLOQUES,
        datos_sinteticos=False,
    )
    (veredicto,) = resultado.skus
    assert veredicto.candidato_campeon == "A/ml/1"
    assert veredicto.razon_sn == pytest.approx(0.7)


def _entrada(sku: str, reales: np.ndarray, observadas: np.ndarray) -> EntradaSku:
    """ML pronostica lo real + 0.2 y DL lo real + 0.5, un dia por origen."""
    return EntradaSku(
        sku=sku,
        sku_class="smooth",
        historia=np.full(30, 10.0),
        reserva=reserva(observadas),
        candidatos=tuple(
            serie(
                f"{sku}/{familia}/{familia}",
                [pronostico(d, [reales[d] + sesgo]) for d in range(100)],
                familia=familia,
                modelo=familia,
            )
            for familia, sesgo in (("ml", 0.2), ("dl", 0.5))
        ),
    )


def test_perturbar_el_bloque_de_prueba_no_cambia_al_campeon() -> None:
    reales = {
        sku: 10 + i + np.sin(np.arange(100) * (1 + i / 10))
        for i, sku in enumerate(("A", "B", "C"))
    }
    # Toda observacion del bloque de prueba (dias 41 a 100) cambia.
    perturbadas = {
        sku: np.r_[valores[:40], 3 * valores[40:] + 5]
        for sku, valores in reales.items()
    }

    def emitir(observadas):
        return emitir_veredictos(
            [
                evaluar_bloques(
                    _entrada(sku, reales[sku], observadas[sku]), corte(100), BLOQUES
                )
                for sku in reales
            ],
            BLOQUES,
            datos_sinteticos=False,
        )

    antes, despues = emitir(reales), emitir(perturbadas)

    def registro(resultado) -> str:
        return json.dumps(
            [asdict(c.seleccion) for c in resultado.categorias], sort_keys=True
        )

    assert registro(despues) == registro(antes)
    assert [(v.familia_campeona, v.candidato_campeon) for v in despues.skus] == [
        ("ml", f"{sku}/ml/ml") for sku in reales
    ]
    # La perturbacion llega al bloque de prueba: ahi DL tendria menor r.
    for veredicto in despues.skus:
        r = {m.familia: m.razon_sn for m in veredicto.modelos_evaluados}
        assert r["dl"] < r["ml"] == pytest.approx(veredicto.razon_sn)
    assert [v.razon_sn for v in despues.skus] != [v.razon_sn for v in antes.skus]


def test_iqr_de_la_diferencia_de_mae_por_ventana() -> None:
    campeon = metricas_candidato("ml", 0.5, n=4, maes=[1.0, 2.0, 3.0, 4.0])
    base = metricas_candidato(FAMILIA_LINEA_BASE, 1.0, n=4, maes=[1.0] * 4)
    resultado = emitir_veredictos(
        [
            en_bloques(
                evaluacion("A", [campeon], base=base),
                seleccion=evaluacion("A", [metricas_candidato("ml", 0.5)]),
            )
        ],
        BLOQUES,
        datos_sinteticos=False,
    )
    # Diferencias [0, 1, 2, 3]: Q3 - Q1 = 2.25 - 0.75.
    assert resultado.skus[0].iqr_diferencia_mae == pytest.approx(1.5)


def test_diebold_mariano_acompana_al_veredicto_sin_cambiarlo() -> None:
    veredicto = _unico(evaluacion("A", [metricas_candidato("ml", 0.8, rmses=_MEJOR)]))[
        "A"
    ]
    prueba = veredicto.diebold_mariano
    esperado = prueba_hln(np.square(_MEJOR) - 1.0, 1)
    assert veredicto.veredicto == "VALIDADO"
    assert prueba.estadistico == pytest.approx(esperado.estadistico)
    # Una sola prueba en la categoria: BH no cambia el p-valor.
    assert prueba.p_ajustado == pytest.approx(esperado.p_valor)
    assert prueba.significativa is True


def test_intervalo_de_r_invierte_la_prueba_del_veredicto() -> None:
    veredicto = _unico(evaluacion("A", [metricas_candidato("ml", 0.8, rmses=_MEJOR)]))[
        "A"
    ]
    intervalo = veredicto.intervalo_r
    assert intervalo is not None and intervalo.superior is not None
    assert intervalo.nivel == pytest.approx(1 - POLITICA_INICIAL.alfa_dm)
    # r de las perdidas por ventana: raiz de la razon de perdidas medias.
    r_ventanas = float(np.sqrt(np.mean(np.square(_MEJOR))))
    assert intervalo.inferior < r_ventanas < intervalo.superior < 1.0
    # Unico SKU significativo de su categoria: nivel FCR 1 - 1 * q / 1.
    assert veredicto.intervalo_r_fcr is not None
    assert veredicto.intervalo_r_fcr.nivel == pytest.approx(0.95)


def test_minimo_dm_configurado_no_cambia_el_veredicto_descriptivo() -> None:
    entrada = evaluacion("A", [metricas_candidato("ml", 0.8, rmses=_MEJOR)])
    resultado = _emitir(entrada, politica=PoliticaSeleccion(n_min_dm=41))
    veredicto = resultado.skus[0]
    assert veredicto.veredicto == "VALIDADO"
    assert veredicto.razon_sn == 0.8
    assert veredicto.diebold_mariano.n_ventanas == 40
    assert veredicto.diebold_mariano.causa == "pocas_ventanas"
    assert veredicto.diebold_mariano.p_valor is None
    assert veredicto.intervalo_r is None
    assert _unico(entrada)["A"].diebold_mariano.significativa is True


def test_sin_diebold_mariano_queda_la_causa() -> None:
    campeon = metricas_candidato("ml", 0.8, rmses=_MEJOR)
    sintetico = _unico(evaluacion("A", [campeon]), datos_sinteticos=True)["A"]
    adverso = _unico(evaluacion("A", [metricas_candidato("ml", 1.3)]))["A"]
    (corto,) = emitir_veredictos(
        [
            en_bloques(
                evaluacion("A", [metricas_candidato("ml", 0.5, n=1)]),
                seleccion=evaluacion("A", [metricas_candidato("ml", 0.5)]),
            )
        ],
        BLOQUES,
        datos_sinteticos=False,
    ).skus
    assert sintetico.diebold_mariano.causa == "datos_sinteticos"
    assert adverso.diebold_mariano.causa == "campeon_es_linea_base"
    assert corto.diebold_mariano.causa == "veredicto:EVIDENCIA_INSUFICIENTE"
    assert all(
        v.diebold_mariano.p_ajustado is None and v.intervalo_r is None
        for v in (sintetico, adverso, corto)
    )


def test_bh_corrige_entre_los_skus_de_cada_categoria() -> None:
    ruido = [1.2, 0.7, 1.1, 0.8, 0.95] * 8
    veredictos = _unico(
        evaluacion("A", [metricas_candidato("ml", 0.8, rmses=_MEJOR)]),
        evaluacion("B", [metricas_candidato("ml", 0.9, rmses=ruido)]),
        evaluacion(
            "C", [metricas_candidato("ml", 0.9, rmses=ruido)], sku_class="erratic"
        ),
    )
    p_a = prueba_hln(np.square(_MEJOR) - 1.0, 1).p_valor
    p_b = prueba_hln(np.square(ruido) - 1.0, 1).p_valor
    assert p_a is not None and p_b is not None and p_a < p_b
    # BH con m=2 en smooth; C es la unica prueba de erratic.
    assert veredictos["A"].diebold_mariano.p_ajustado == pytest.approx(
        min(2 * p_a, p_b)
    )
    assert veredictos["B"].diebold_mariano.p_ajustado == pytest.approx(p_b)
    assert veredictos["C"].diebold_mariano.p_ajustado == pytest.approx(p_b)


def test_significancia_por_categoria_con_bh_by_y_nivel_fcr() -> None:
    deltas = (0.30, 0.12, 0.076, 0.074, 0.07, 0.02)
    resultado = _emitir(
        *(
            evaluacion(f"S{k}", [metricas_candidato("ml", 0.9, rmses=_ruido(d, k))])
            for k, d in enumerate(deltas)
        )
    )
    p = [
        prueba_hln(np.square(_ruido(d, k)) - 1.0, 1).p_valor
        for k, d in enumerate(deltas)
    ]
    bh = multipletests(p, alpha=0.05, method="fdr_bh")[0]
    by = multipletests(p, alpha=0.05, method="fdr_by")[0]
    # BY es mas estricto: aqui rechaza un SKU menos que BH.
    assert (int(bh.sum()), int(by.sum())) == (5, 4)
    significancia = resultado.categorias[0].significancia
    assert significancia.n_probados == 6
    assert (significancia.n_mejores, significancia.n_peores) == (5, 0)
    assert (significancia.n_mejores_by, significancia.n_peores_by) == (4, 0)
    assert significancia.nivel_fcr == pytest.approx(1 - 5 * 0.05 / 6)
    for veredicto, rechazo in zip(resultado.skus, bh, strict=True):
        assert veredicto.diebold_mariano.significativa is bool(rechazo)
        fcr = veredicto.intervalo_r_fcr
        assert (fcr is not None) is bool(rechazo)
        if fcr is not None:
            assert fcr.nivel == pytest.approx(significancia.nivel_fcr)
            assert veredicto.intervalo_r is not None
            # Mayor nivel, intervalo mas ancho.
            assert fcr.inferior <= veredicto.intervalo_r.inferior


def test_sin_pruebas_la_significancia_no_tiene_nivel_fcr() -> None:
    resultado = _emitir(
        evaluacion("A", [metricas_candidato("ml", 0.8, rmses=_MEJOR)]),
        datos_sinteticos=True,
    )
    significancia = resultado.categorias[0].significancia
    assert (significancia.n_probados, significancia.n_mejores) == (0, 0)
    assert significancia.nivel_fcr is None


def test_confirmacion_del_campeon_congelado_por_categoria() -> None:
    resultado = _emitir(
        *(
            evaluacion(
                f"S{k}",
                [
                    metricas_candidato("ml", 0.7, rmses=_ruido(0.3, k)),
                    metricas_candidato("dl", 0.9, rmses=_ruido(0.05, k)),
                ],
            )
            for k in range(12)
        )
    )
    confirmacion = resultado.categorias[0].confirmacion
    assert confirmacion.familia_campeona == "ml"
    assert confirmacion.estado == "confirmado"
    assert [c.competidora for c in confirmacion.comparaciones] == ["dl"]
    assert confirmacion.comparaciones[0].n_skus == 12
    assert confirmacion.p_holm == pytest.approx(confirmacion.p_iut)
    assert confirmacion.efecto_linea_base is not None
    assert confirmacion.efecto_linea_base.n_skus == 12


def test_contendiente_es_la_mejor_familia_sin_la_fundacional() -> None:
    resultado = _emitir(
        evaluacion(
            "A",
            [
                metricas_candidato("foundation", 0.6),
                metricas_candidato("ml", 0.7),
                metricas_candidato("classical", 0.9),
            ],
        ),
        evaluacion(
            "B",
            [metricas_candidato("ml", 0.8), metricas_candidato("classical", 0.7)],
            sku_class="erratic",
        ),
    )
    erratic, smooth = resultado.categorias
    assert smooth.seleccion.familia_campeona == "foundation"
    assert smooth.contendiente == "ml"
    assert erratic.contendiente is None


def test_seleccion_previa_aplica_la_regla_a_los_pliegues_l3() -> None:
    sin_previas = _emitir(evaluacion("A", [metricas_candidato("ml", 0.8)]))
    assert sin_previas.categorias[0].seleccion_previa is None
    previa = evaluacion(
        "A", [metricas_candidato("ml", 0.95), metricas_candidato("dl", 0.85)]
    )
    con_previas = _emitir(
        evaluacion("A", [metricas_candidato("ml", 0.8), metricas_candidato("dl", 0.9)]),
        seleccion_previa=seleccionar_previa([previa], ["smooth"]),
    )
    smooth = con_previas.categorias[0]
    assert smooth.seleccion.familia_campeona == "ml"
    assert smooth.seleccion_previa is not None
    assert smooth.seleccion_previa.familia_campeona == "dl"
    assert smooth.seleccion_previa.medianas_r == {"dl": 0.85, "ml": 0.95}


def test_resumen_por_categoria() -> None:
    resultado = _emitir(
        evaluacion("A", [metricas_candidato("ml", 0.5)]),
        evaluacion("B", [metricas_candidato("ml", 1.2)]),
        evaluacion("C", [metricas_candidato("ml", 0.7)], n_obs=0),
        evaluacion("D", [metricas_candidato("ml", 0.9)], sku_class="erratic"),
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
    # Una sola familia: no hay competidoras que confirmar.
    assert smooth.confirmacion.estado == "sin_comparables"
    assert smooth.confirmacion.causa == "sin_familias_competidoras"
    assert resultado.version_politica == POLITICA_INICIAL.version
    assert resultado.version_metricas == VERSION_METRICAS
    assert resultado.datos_sinteticos is False


def test_sku_repetido_se_rechaza() -> None:
    with pytest.raises(EvaluacionRetrospectivaError):
        _emitir(evaluacion("A", []), evaluacion("A", []))


def test_bloques_con_otra_fraccion_que_la_politica_se_rechazan() -> None:
    otros = BloquesReserva(BLOQUES.t_star, BLOQUES.dias_reserva, 1, 0.5)
    with pytest.raises(EvaluacionRetrospectivaError, match="fraccion"):
        emitir_veredictos(
            [en_bloques(evaluacion("A", []))], otros, datos_sinteticos=False
        )


def test_flujo_de_metricas_a_veredicto_sobre_el_bloque_de_prueba() -> None:
    # Integracion 3.4-A1 -> A2: 100 ventanas de un dia, candidato mejor que SN.
    reales = 10 + np.sin(np.arange(100))
    entrada = EntradaSku(
        sku="A",
        sku_class="smooth",
        historia=np.full(30, 10.0),
        reserva=reserva(reales),
        candidatos=(
            serie(
                "A/ml/lightgbm",
                [pronostico(d, [reales[d] + 0.1]) for d in range(100)],
                familia="ml",
                modelo="lightgbm",
            ),
        ),
    )
    evaluado = evaluar_bloques(entrada, corte(100), BLOQUES)
    (veredicto,) = emitir_veredictos([evaluado], BLOQUES, datos_sinteticos=False).skus
    # Seasonal Naive (m=7): 10 desde la historia, luego lo observado 7 dias antes.
    linea_base = np.array([10.0 if d < 7 else reales[d - 7] for d in range(100)])
    # Bloque de prueba: origenes t*+40 .. t*+99, dias 41 a 100.
    rmse_sn = np.sqrt(np.mean((reales[40:] - linea_base[40:]) ** 2))
    assert veredicto.veredicto == "VALIDADO"
    assert veredicto.n_ventanas == 60
    assert veredicto.razon_sn == pytest.approx(0.1 / rmse_sn)
    assert veredicto.diebold_mariano.n_ventanas == 60
    assert veredicto.diebold_mariano.horizonte == 1
    assert veredicto.diebold_mariano.significativa is True
