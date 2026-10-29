"""Persistencia 3.5-A1: identidad, idempotencia, atomicidad y estados (ADR-03-010)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from tests._evaluaciones import T_ESTRELLA, fechas_desde, pronostico, serie
from tests._evidencia import (
    INGESTA,
    base,
    corrida,
    corrida_evaluada,
    fallo,
    filas,
    handoff,
    identidad,
    observaciones,
)

from pred_engine.forecasting.adaptador_candidatos import LoteRechazadoError
from pred_engine.forecasting.evaluaciones.calculo_errores import (
    PronosticoFechado,
    SerieCandidato,
)
from pred_engine.forecasting.evaluaciones.confirmacion import (
    ComparacionFamilia,
    Intervalo,
)
from pred_engine.forecasting.evaluaciones.diebold_mariano import IntervaloRazon
from pred_engine.forecasting.evaluaciones.veredictos import SignificanciaCategoria
from pred_engine.forecasting.persistencia_evidencia import (
    VENTANA_AGREGADA,
    EstadoCorridaError,
    EvidenciaInconsistenteError,
    consultar,
    estado_corrida,
    guardar_evaluacion,
    guardar_unidades,
    marcar_evaluada,
    registrar_corrida,
    unidades_confirmadas,
)

_SARIMA = "S1/classical/sarima"


def _sarima(*valores: list[float]) -> SerieCandidato:
    """Una ventana por lista de valores, con origenes t*, t*+2, ..."""
    return serie(_SARIMA, [pronostico(2 * i, v) for i, v in enumerate(valores)])


# --- Identidad y registro ----------------------------------------------------


def test_la_misma_identidad_da_el_mismo_run_id_y_otra_politica_otro() -> None:
    assert identidad().run_id == identidad().run_id
    assert identidad().run_id.startswith("m3-")
    assert identidad(version_politica="3.4.1").run_id != identidad().run_id


def test_registrar_deja_la_corrida_en_curso_con_sus_fallos() -> None:
    conn = base()
    run_id = corrida(conn, fallo())
    assert estado_corrida(conn, run_id) == "EN_CURSO"
    run_id_m2, guardada = conn.execute(
        "SELECT run_id_m2, identidad FROM m3_corridas"
    ).fetchone()
    assert run_id_m2 == "m2-corrida-1"
    assert json.loads(guardada)["t_corte_reserva"] == "2024-01-14"
    assert filas(conn, "m3_fallos") == 1
    # SKUs del manifiesto: los de los candidatos validados y los de los fallos.
    assert consultar(conn, run_id)["m3_skus"]["sku"].tolist() == ["S1", "S2"]


def test_registrar_otra_vez_la_misma_identidad_no_cambia_nada() -> None:
    conn = base()
    run_id = corrida(conn, fallo())
    assert corrida(conn, fallo(), fallo("S2")) == run_id
    assert filas(conn, "m3_corridas") == 1
    assert filas(conn, "m3_fallos") == 1


def test_un_lote_rechazado_queda_con_su_causa_y_no_admite_unidades() -> None:
    conn = base()
    run_id = registrar_corrida(
        conn,
        identidad(),
        LoteRechazadoError("contexto_incompatible", "otra particion"),
        ingesta_sha256=INGESTA,
        datos_sinteticos=False,
    )
    assert estado_corrida(conn, run_id) == "RECHAZADA"
    (causa,) = conn.execute("SELECT causa_rechazo FROM m3_corridas").fetchone()
    assert causa == "contexto_incompatible: otra particion"
    assert filas(conn, "m3_skus") == 0
    with pytest.raises(EstadoCorridaError, match="RECHAZADA"):
        guardar_unidades(conn, run_id, "S1", _sarima([1.0]))


def test_la_ingesta_debe_existir_en_la_plataforma() -> None:
    conn = base()
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        registrar_corrida(
            conn,
            identidad(),
            handoff(),
            ingesta_sha256="d" * 64,
            datos_sinteticos=False,
        )
    assert filas(conn, "m3_corridas") == 0


# --- Unidades: idempotencia y atomicidad -------------------------------------


def test_cada_origen_es_una_unidad_con_sus_pronosticos() -> None:
    conn = base()
    run_id = corrida(conn)
    nuevas = guardar_unidades(conn, run_id, "S1", _sarima([1.0, np.nan], [3.0, 4.0]))
    assert nuevas == 2
    pronosticos = consultar(conn, run_id)["m3_pronosticos"]
    assert pronosticos["origen"].tolist() == ["2024-01-14"] * 2 + ["2024-01-16"] * 2
    assert pronosticos["h"].tolist() == [1, 2, 1, 2]
    assert pronosticos["fecha"].tolist()[:2] == ["2024-01-15", "2024-01-16"]
    # Un paso sin pronostico finito se guarda como NULL, no se descarta.
    assert pronosticos["valor"].isna().tolist() == [False, True, False, False]


def test_reinsertar_una_unidad_confirmada_no_duplica_ni_cambia_valores() -> None:
    conn = base()
    run_id = corrida(conn)
    guardar_unidades(conn, run_id, "S1", _sarima([1.0, 2.0]))
    assert guardar_unidades(conn, run_id, "S1", _sarima([7.0, 7.0, 7.0])) == 0
    assert consultar(conn, run_id)["m3_pronosticos"]["valor"].tolist() == [1.0, 2.0]
    assert filas(conn, "m3_unidades") == 1


def test_un_sku_fuera_del_manifiesto_no_admite_unidades() -> None:
    conn = base()
    run_id = corrida(conn)
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        guardar_unidades(
            conn, run_id, "S9", serie("S9/classical/sarima", [pronostico(0, [1.0])])
        )
    assert filas(conn, "m3_unidades") == 0


def test_un_pronostico_de_forma_invalida_se_guarda_sin_valores() -> None:
    conn = base()
    run_id = corrida(conn)
    mal_formado = PronosticoFechado(
        origen=T_ESTRELLA,
        fechas=fechas_desde(T_ESTRELLA, 2),
        valores=np.array([1.0]),
    )
    guardar_unidades(conn, run_id, "S1", serie(_SARIMA, [mal_formado]))
    valores = consultar(conn, run_id)["m3_pronosticos"]["valor"]
    assert len(valores) == 2 and valores.isna().all()


def test_una_falla_a_mitad_de_la_transaccion_no_deja_marcador_ni_pronosticos() -> None:
    conn = base()
    run_id = corrida(conn)
    conn.execute(
        "CREATE TRIGGER falla BEFORE INSERT ON m3_pronosticos WHEN NEW.h = 2"
        " BEGIN SELECT RAISE(ABORT, 'falla simulada'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="falla simulada"):
        guardar_unidades(conn, run_id, "S1", _sarima([1.0], [2.0, 3.0]))
    assert filas(conn, "m3_unidades") == 0
    assert filas(conn, "m3_pronosticos") == 0


def test_unidades_confirmadas_permiten_reanudar_sin_recalcular() -> None:
    conn = base()
    run_id = corrida(conn)
    guardar_unidades(conn, run_id, "S1", _sarima([1.0], [2.0]))
    assert unidades_confirmadas(conn, run_id) == {
        ("S1", _SARIMA, T_ESTRELLA),
        ("S1", _SARIMA, T_ESTRELLA + pd.Timedelta(days=2)),
    }


def test_no_se_escribe_en_una_corrida_no_registrada(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errores: list[str] = []

    class _Captura:
        def info(self, *args: object) -> None:
            pass

        def error(self, mensaje: str, *args: object) -> None:
            errores.append(mensaje % args)

    monkeypatch.setattr(
        "pred_engine.forecasting.persistencia_evidencia.repositorio._logger",
        _Captura(),
    )
    with pytest.raises(EstadoCorridaError, match="no esta registrada"):
        guardar_unidades(base(), "m3-inexistente", "S1", _sarima([1.0]))
    assert len(errores) == 1 and "operacion=guardar_unidades" in errores[0]


# --- Evaluacion: metricas, seleccion y veredictos ----------------------------


def test_marcar_evaluada_es_idempotente_y_cierra_las_unidades() -> None:
    conn = base()
    run_id = corrida(conn)
    marcar_evaluada(conn, run_id)
    marcar_evaluada(conn, run_id)
    assert estado_corrida(conn, run_id) == "EVALUADA"
    with pytest.raises(EstadoCorridaError, match="EVALUADA"):
        guardar_unidades(conn, run_id, "S1", _sarima([1.0]))


def test_la_evaluacion_exige_la_corrida_evaluada() -> None:
    conn = base()
    run_id = corrida(conn)
    _, evaluaciones, resultado = corrida_evaluada(base())
    with pytest.raises(EstadoCorridaError, match="EN_CURSO"):
        guardar_evaluacion(
            conn,
            run_id,
            evaluaciones,
            resultado,
            observaciones=observaciones(conn, run_id),
        )


def test_guardar_evaluacion_persiste_metricas_seleccion_y_veredictos() -> None:
    conn = base()
    run_id, evaluaciones, resultado = corrida_evaluada(conn)
    guardar_evaluacion(
        conn, run_id, evaluaciones, resultado, observaciones=observaciones(conn, run_id)
    )
    assert estado_corrida(conn, run_id) == "SELECCIONADA"

    metricas = consultar(conn, run_id, sku="S1")["m3_metricas"]
    sarima = metricas[metricas["candidato_id"] == _SARIMA].set_index(
        ["ventana", "metrica"]
    )["valor"]
    # Real 10, SARIMA 9 y Seasonal Naive 6: MAE = 1 y r = 1 / 4. Las ventanas son
    # las de toda la reserva; la agregada, la del bloque de prueba (origen t*+2).
    assert sarima[("2024-01-14", "mae")] == pytest.approx(1.0)
    assert sarima[("2024-01-16", "mae")] == pytest.approx(1.0)
    assert sarima[(VENTANA_AGREGADA, "razon_sn")] == pytest.approx(0.25)
    assert sarima[(VENTANA_AGREGADA, "n_ventanas_validas")] == 1

    categorias = consultar(conn, run_id)["m3_categorias"].set_index("sku_class")
    assert set(json.loads(categorias.loc["smooth", "medianas_r"])) == {
        "classical",
        "ml",
    }
    # En lumpy solo compite la familia elegible que M2 entrego (ADR-03-003).
    assert set(json.loads(categorias.loc["lumpy", "medianas_r"])) == {"classical"}
    assert json.loads(categorias.loc["lumpy", "familias_excluidas"]) == {
        "foundation": "no_entregada",
        "ml": "no_elegible",
    }

    veredicto = consultar(conn, run_id, sku="S1")["m3_veredictos"].iloc[0]
    assert veredicto["veredicto"] == "VALIDADO"
    assert veredicto["candidato_campeon"] == _SARIMA
    evaluados = json.loads(veredicto["modelos_evaluados"])
    assert [m["familia"] for m in evaluados] == ["seasonal_naive", "classical", "ml"]
    assert evaluados[0]["razon_sn"] == 1.0
    # Una ventana de prueba no alcanza n_min_dm: no hay prueba y queda la causa.
    assert veredicto["dm_causa"] == "pocas_ventanas"
    assert pd.isna(veredicto["dm_p_valor"])
    # Sin p-valores ni competidoras con SKU suficientes: resumen sin inferencia.
    significancia = consultar(conn, run_id)["m3_significancia"].set_index("sku_class")
    assert significancia.loc["smooth", "n_probados"] == 0
    assert pd.isna(significancia.loc["smooth", "nivel_fcr"])
    confirmaciones = consultar(conn, run_id)["m3_confirmaciones"].set_index("sku_class")
    assert confirmaciones.loc["smooth", "estado"] == "sin_comparables"
    assert confirmaciones.loc["smooth", "causa"] == "pocos_skus:ml"
    assert filas(conn, "m3_intervalos_r") == filas(conn, "m3_seleccion_previa") == 0


def test_guardar_evaluacion_persiste_la_evidencia_del_bloque_de_prueba() -> None:
    conn = base()
    run_id, evaluaciones, resultado = corrida_evaluada(conn)
    lumpy, smooth = resultado.categorias
    fila = ComparacionFamilia(
        "ml",
        12,
        0.8,
        Intervalo(0.7, 0.9),
        0.79,
        Intervalo(0.72, 0.88),
        11,
        Intervalo(0.65, 0.98),
        3.0,
        0.001,
        0.001,
        5,
        0,
        None,
    )
    smooth = replace(
        smooth,
        significancia=SignificanciaCategoria(4, 3, 1, 2, 1, 0.95),
        confirmacion=replace(
            smooth.confirmacion,
            estado="confirmado",
            causa=None,
            p_iut=0.001,
            p_holm=0.002,
            comparaciones=(fila,),
        ),
        contendiente="ml",
        seleccion_previa=smooth.seleccion,
    )
    skus = tuple(
        replace(
            v,
            intervalo_r=IntervaloRazon(0.95, 0.2, None),
            intervalo_r_fcr=IntervaloRazon(0.96, 0.19, None),
        )
        if v.sku == "S1"
        else v
        for v in resultado.skus
    )
    resultado = replace(resultado, categorias=(lumpy, smooth), skus=skus)
    guardar_evaluacion(
        conn, run_id, evaluaciones, resultado, observaciones=observaciones(conn, run_id)
    )
    evidencia = consultar(conn, run_id)
    confirmacion = evidencia["m3_confirmaciones"].set_index("sku_class").loc["smooth"]
    assert (confirmacion["estado"], confirmacion["p_holm"]) == ("confirmado", 0.002)
    assert confirmacion["contendiente"] == "ml"
    comparacion = evidencia["m3_comparaciones"].iloc[0]
    assert comparacion["competidora"] == "ml"
    assert (
        comparacion["ic_hodges_lehmann_inferior"],
        comparacion["ic_hodges_lehmann_superior"],
    ) == (0.72, 0.88)
    assert pd.isna(comparacion["causa"])
    significancia = evidencia["m3_significancia"].set_index("sku_class").loc["smooth"]
    assert significancia[["n_mejores", "n_peores_by", "nivel_fcr"]].tolist() == [
        3,
        1,
        0.95,
    ]
    previa = evidencia["m3_seleccion_previa"].iloc[0]
    assert (previa["sku_class"], previa["familia_campeona"]) == ("smooth", "classical")
    assert json.loads(previa["medianas_r"]) == dict(smooth.seleccion.medianas_r)
    intervalo = consultar(conn, run_id, sku="S1")["m3_intervalos_r"].iloc[0]
    assert (intervalo["nivel"], intervalo["inferior"]) == (0.95, 0.2)
    assert pd.isna(intervalo["superior"]) and pd.isna(intervalo["superior_fcr"])
    assert intervalo["nivel_fcr"] == 0.96


def test_los_pronosticos_de_la_linea_base_tambien_se_guardan() -> None:
    conn = base()
    run_id, _, _ = corrida_evaluada(conn)
    unidades = consultar(conn, run_id, sku="S1")["m3_unidades"]
    assert "S1/seasonal_naive/seasonal_naive" in set(unidades["candidato_id"])


def test_una_metrica_no_calculable_queda_nula_con_su_causa() -> None:
    conn = base()
    run_id, evaluaciones, resultado = corrida_evaluada(conn)
    s1 = evaluaciones[0]
    # La fila agregada se escribe desde el bloque de prueba.
    prueba = s1.prueba
    assert prueba.linea_base.agregadas is not None
    sin_escala = replace(
        prueba.linea_base.agregadas,
        mase=None,
        no_calculables={"mase": "historia_corta"},
    )
    evaluaciones[0] = replace(
        s1,
        prueba=replace(
            prueba, linea_base=replace(prueba.linea_base, agregadas=sin_escala)
        ),
    )
    guardar_evaluacion(
        conn, run_id, evaluaciones, resultado, observaciones=observaciones(conn, run_id)
    )
    metricas = consultar(conn, run_id, sku="S1")["m3_metricas"]
    mase = metricas[
        (metricas["candidato_id"] == prueba.linea_base.candidato_id)
        & (metricas["ventana"] == VENTANA_AGREGADA)
        & (metricas["metrica"] == "mase")
    ].iloc[0]
    assert pd.isna(mase["valor"]) and mase["causa"] == "historia_corta"


def test_guardar_la_evaluacion_dos_veces_no_duplica() -> None:
    conn = base()
    run_id, evaluaciones, resultado = corrida_evaluada(conn)
    guardar_evaluacion(
        conn, run_id, evaluaciones, resultado, observaciones=observaciones(conn, run_id)
    )
    antes = {t: filas(conn, t) for t in ("m3_metricas", "m3_veredictos")}
    guardar_evaluacion(
        conn, run_id, evaluaciones, resultado, observaciones=observaciones(conn, run_id)
    )
    assert {t: filas(conn, t) for t in antes} == antes


def test_versiones_distintas_a_las_de_la_identidad_se_rechazan_sin_escribir() -> None:
    conn = base()
    run_id, evaluaciones, resultado = corrida_evaluada(conn)
    otra = replace(resultado, version_politica="otra")
    with pytest.raises(EvidenciaInconsistenteError, match="versiones"):
        guardar_evaluacion(
            conn, run_id, evaluaciones, otra, observaciones=observaciones(conn, run_id)
        )
    assert estado_corrida(conn, run_id) == "EVALUADA"
    assert filas(conn, "m3_metricas") == 0


def test_sin_los_pronosticos_de_la_linea_base_no_se_guarda_la_evaluacion() -> None:
    conn = base()
    run_id, evaluaciones, resultado = corrida_evaluada(conn, con_linea_base=False)
    with pytest.raises(EvidenciaInconsistenteError, match="seasonal_naive"):
        guardar_evaluacion(
            conn,
            run_id,
            evaluaciones,
            resultado,
            observaciones=observaciones(conn, run_id),
        )
    assert estado_corrida(conn, run_id) == "EVALUADA"


def test_veredictos_con_otro_origen_de_datos_que_la_corrida_se_rechazan() -> None:
    conn = base()
    run_id, evaluaciones, resultado = corrida_evaluada(conn)
    sintetico = replace(resultado, datos_sinteticos=True)
    with pytest.raises(EvidenciaInconsistenteError, match="datos_sinteticos"):
        guardar_evaluacion(
            conn,
            run_id,
            evaluaciones,
            sintetico,
            observaciones=observaciones(conn, run_id),
        )
    assert filas(conn, "m3_veredictos") == 0


@pytest.mark.parametrize(
    ("veredicto", "dm_causa"),
    [
        ("mantiene", "datos_sinteticos"),
        # Diebold-Mariano sin p-valor necesita su causa.
        ("VALIDADO", None),
    ],
)
def test_un_veredicto_fuera_de_las_reglas_viola_el_esquema(
    veredicto: str, dm_causa: str | None
) -> None:
    conn = base()
    run_id = corrida(conn)
    conn.execute(
        "INSERT INTO m3_categorias VALUES (?, 'smooth', 'ml', 'menor_mediana', 0,"
        " '{}', '[]', '{}', '{}', '{}', '{}', NULL, 0)",
        (run_id,),
    )
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        conn.execute(
            "INSERT INTO m3_veredictos (run_id, sku, sku_class, veredicto,"
            " familia_campeona, n_ventanas, cobertura, comparacion_incompleta,"
            " dm_n_ventanas, dm_horizonte, dm_causa, justificacion,"
            " modelos_evaluados) VALUES (?, 'S1', 'smooth', ?, 'ml', 0, 0.0, 0,"
            " 0, 0, ?, '', '[]')",
            (run_id, veredicto, dm_causa),
        )


def test_consultar_por_sku_solo_devuelve_ese_sku() -> None:
    conn = base()
    run_id, evaluaciones, resultado = corrida_evaluada(conn, fallo("S2"))
    guardar_evaluacion(
        conn, run_id, evaluaciones, resultado, observaciones=observaciones(conn, run_id)
    )
    por_sku = consultar(conn, run_id, sku="S2")
    assert "m3_corridas" not in por_sku
    # Sin p-valores en esta corrida no hay intervalos de r.
    assert por_sku.pop("m3_intervalos_r").empty
    assert all(set(frame["sku"]) == {"S2"} for frame in por_sku.values())


def test_guardar_evaluacion_publica_la_reserva_real_con_nulos() -> None:
    conn = base()
    run_id, evaluaciones, resultado = corrida_evaluada(conn)
    reales = observaciones(conn, run_id)
    reales["S2"] = reales["S2"].where(reales["S2"].index != reales["S2"].index[1])
    guardar_evaluacion(conn, run_id, evaluaciones, resultado, observaciones=reales)
    assert conn.execute(
        "SELECT fecha, valor FROM m3_observaciones WHERE sku = 'S2' ORDER BY fecha"
    ).fetchall() == [
        ("2024-01-15", 10.0),
        ("2024-01-16", None),
        ("2024-01-17", 10.0),
        ("2024-01-18", 10.0),
    ]
    assert filas(conn, "m3_observaciones") == 8


@pytest.mark.parametrize(
    ("recortar", "mensaje"),
    [
        (lambda o: {"S1": o["S1"]}, "SKU del manifiesto"),
        (lambda o: o | {"S2": o["S2"].iloc[:-1]}, "sin fila de demanda observada"),
        (
            lambda o: o | {"S2": pd.Series([1.0], index=[T_ESTRELLA])},
            "fuera de la reserva",
        ),
    ],
)
def test_la_demanda_observada_incompleta_no_guarda_la_evaluacion(
    recortar, mensaje
) -> None:
    conn = base()
    run_id, evaluaciones, resultado = corrida_evaluada(conn)
    with pytest.raises(EvidenciaInconsistenteError, match=mensaje):
        guardar_evaluacion(
            conn,
            run_id,
            evaluaciones,
            resultado,
            observaciones=recortar(observaciones(conn, run_id)),
        )
    assert estado_corrida(conn, run_id) == "EVALUADA"
    assert filas(conn, "m3_observaciones") == filas(conn, "m3_veredictos") == 0


def test_sin_corte_fijado_no_se_guarda_la_evaluacion() -> None:
    conn = base()
    run_id, evaluaciones, resultado = corrida_evaluada(conn)
    conn.execute("DELETE FROM m3_cortes")
    conn.commit()
    with pytest.raises(EvidenciaInconsistenteError, match="sin corte"):
        guardar_evaluacion(
            conn,
            run_id,
            evaluaciones,
            resultado,
            observaciones=observaciones(conn, run_id),
        )
    assert estado_corrida(conn, run_id) == "EVALUADA"
