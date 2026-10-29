"""Protocolo congelado, fallos por ventana y lectura segura de unidades M3."""

import sqlite3
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from tests._evidencia import base, corrida, handoff, identidad
from tests._manifiestos import manifiesto

from pred_engine.comun.errores_codigo import ELEGIBILIDAD_MENOS_DE_DOS_FAMILIAS
from pred_engine.comun.exclusiones import SkuExclusion
from pred_engine.comun.reserva import ReserveCut
from pred_engine.forecasting.base import SeasonalNaiveStub
from pred_engine.forecasting.generador_pronosticos import (
    ConfiguracionPronosticos,
    generar_pronosticos,
)
from pred_engine.forecasting.persistencia_evidencia import (
    EvidenciaInconsistenteError,
    marcar_evaluada,
    registrar_corrida,
)
from pred_engine.forecasting.persistencia_evidencia.recuperacion import (
    candidatos_con_fallos,
    cargar_unidades,
    fijar_protocolo,
    guardar_unidades_generadas,
)

_CORTE = {
    "t_star": pd.Timestamp("2024-01-14"),
    "lanzamientos_tardios": (),
    "exclusiones": (),
    "clases": {},
}


def _preparar(**fijar):
    conn = base()
    run_id = corrida(conn)
    h = handoff()
    texto = manifiesto(
        *(c.model_dump(mode="json") for c in h.candidatos),
        contexto=h.contexto.model_dump(mode="json"),
    )
    cfg = ConfiguracionPronosticos(2, identidad().codigo_version)
    protocolo = {"pronosticos": {"horizonte": 2}, "politica": "fijada"}
    fijar_protocolo(
        conn,
        run_id,
        protocolo=protocolo,
        manifiesto=texto,
        procedencia={},
        reserva_abierta_en=datetime.now(UTC),
        n_unidades_previstas=18,
        **_CORTE | fijar,
    )
    corte = ReserveCut(
        pd.Timestamp("2024-01-14"),
        pd.Timestamp("2024-01-15"),
        pd.Timestamp("2024-01-18"),
        4,
    )
    datos = pd.Series(
        np.arange(1.0, 19.0), index=pd.date_range("2024-01-01", periods=18)
    )
    return conn, run_id, h.candidatos[0], cfg, corte, datos, protocolo, texto


def test_pronosticos_y_metadatos_se_restauran_sin_recalcular():
    conn, rid, c, cfg, corte, datos, _, _ = _preparar()
    generar_pronosticos(
        c,
        datos,
        corte,
        cfg,
        fabricas={"classical": lambda config, *, seed: SeasonalNaiveStub(1, seed)},
        confirmar=lambda u: guardar_unidades_generadas(conn, rid, [u]),
    )
    confirmadas = cargar_unidades(conn, rid, c)
    assert [u.pronostico.valores.tolist() for u in confirmadas] == [
        [14.0] * 2,
        [15.0] * 2,
        [16.0] * 2,
    ]
    assert confirmadas[0].codigo_version == "0123456789abcdef"
    assert confirmadas[0].candidato.semilla == 7
    assert candidatos_con_fallos(conn, rid, "S1") == frozenset()
    with pytest.raises(EvidenciaInconsistenteError, match="unidades"):
        marcar_evaluada(conn, rid)
    assert conn.execute("SELECT estado FROM m3_corridas").fetchone()[0] == "EN_CURSO"
    conn.close()


def test_fallo_parcial_durable_cuenta_como_comparacion_incompleta():
    class FallaInicial(SeasonalNaiveStub):
        def pronosticar(self, observadas, horizon):
            return (
                np.full(horizon, np.nan)
                if not len(observadas)
                else super().pronosticar(observadas, horizon)
            )

    conn, rid, c, cfg, corte, datos, _, _ = _preparar()
    generar_pronosticos(
        c,
        datos,
        corte,
        cfg,
        fabricas={"classical": lambda config, *, seed: FallaInicial(1, seed)},
        confirmar=lambda u: guardar_unidades_generadas(conn, rid, [u]),
    )
    restauradas = cargar_unidades(conn, rid, c)
    assert candidatos_con_fallos(conn, rid, "S1") == frozenset({"S1/classical/sarima"})
    assert restauradas[0].fallo.codigo_error == "pronostico_invalido"
    assert np.isnan(restauradas[0].pronostico.valores).all()
    assert restauradas[-1].pronostico.valores.tolist() == [16.0, 16.0]

    def imposible(config, *, seed):
        raise KeyboardInterrupt("no debe reintentar unidades durables")

    reanudada = generar_pronosticos(
        c, datos, corte, cfg, fabricas={"classical": imposible}, confirmadas=restauradas
    )
    assert len(reanudada.fallos) == 1
    assert reanudada.serie.pronosticos[-1].valores.tolist() == [16.0, 16.0]
    conn.close()


def test_fila_de_pronostico_perdida_se_rechaza_sin_inventar_reemplazo():
    conn, rid, c, cfg, corte, datos, _, _ = _preparar()
    generar_pronosticos(
        c,
        datos,
        corte,
        cfg,
        fabricas={"classical": lambda config, *, seed: SeasonalNaiveStub(1, seed)},
        confirmar=lambda u: guardar_unidades_generadas(conn, rid, [u]),
    )
    conn.execute("DELETE FROM m3_pronosticos WHERE origen = '2024-01-14' AND h = 2")
    conn.commit()
    with pytest.raises(EvidenciaInconsistenteError, match="incompleta"):
        cargar_unidades(conn, rid, c)
    assert conn.execute("SELECT COUNT(*) FROM m3_pronosticos").fetchone()[0] == 5
    conn.close()


def test_protocolo_distinto_no_reemplaza_el_anterior():
    conn, rid, _, _, _, _, protocolo, texto = _preparar()
    with pytest.raises(EvidenciaInconsistenteError, match="incompatible"):
        fijar_protocolo(
            conn,
            rid,
            protocolo={**protocolo, "politica": "otra"},
            manifiesto=texto,
            procedencia={},
            reserva_abierta_en=datetime.now(UTC),
            n_unidades_previstas=18,
            **_CORTE,
        )
    assert conn.execute("SELECT COUNT(*) FROM m3_protocolos").fetchone()[0] == 1
    assert (
        '"politica":"fijada"'
        in conn.execute("SELECT protocolo FROM m3_protocolos").fetchone()[0]
    )
    conn.close()


def test_no_se_puede_recualificar_una_ingesta_sintetica_como_real():
    conn = base()
    registrar_corrida(
        conn, identidad(), handoff(), ingesta_sha256="c" * 64, datos_sinteticos=True
    )
    with pytest.raises(EvidenciaInconsistenteError, match="origen"):
        registrar_corrida(
            conn,
            identidad(configuracion_id="otra"),
            handoff(),
            ingesta_sha256="c" * 64,
            datos_sinteticos=False,
        )
    assert conn.execute("SELECT datos_sinteticos FROM m3_corridas").fetchall() == [(1,)]
    conn.close()


def _fijar_de_nuevo(conn, rid, protocolo, texto, **corte):
    fijar_protocolo(
        conn,
        rid,
        protocolo=protocolo,
        manifiesto=texto,
        procedencia={},
        reserva_abierta_en=datetime.now(UTC),
        n_unidades_previstas=18,
        **_CORTE | corte,
    )


def test_el_corte_fijado_no_cambia_ni_admite_otra_causa():
    conn, rid, _, _, _, _, protocolo, texto = _preparar()
    _fijar_de_nuevo(conn, rid, protocolo, texto)
    with pytest.raises(EvidenciaInconsistenteError, match="corte"):
        _fijar_de_nuevo(conn, rid, protocolo, texto, t_star=pd.Timestamp("2024-01-15"))
    tardio = SkuExclusion("TARDE", "lanzado despues del corte")
    with pytest.raises(EvidenciaInconsistenteError, match="lanzamientos tardios"):
        _fijar_de_nuevo(conn, rid, protocolo, texto, lanzamientos_tardios=(tardio,))
    with pytest.raises(EvidenciaInconsistenteError, match="otra causa"):
        _fijar_de_nuevo(
            conn,
            rid,
            protocolo,
            texto,
            lanzamientos_tardios=(SkuExclusion("TARDE", "sin historia admisible"),),
        )
    assert conn.execute("SELECT run_id, t_star FROM m3_cortes").fetchall() == [
        (rid, "2024-01-14")
    ]
    assert conn.execute("SELECT COUNT(*) FROM m3_lanzamientos_tardios").fetchone() == (
        0,
    )
    conn.close()


def test_las_exclusiones_de_m2_se_fijan_una_vez_con_su_categoria():
    causa = "menos de dos familias evaluables tras L2: evaluable ml"
    excluido = SkuExclusion("X", causa, ELEGIBILIDAD_MENOS_DE_DOS_FAMILIAS)
    conn, rid, _, _, _, _, protocolo, texto = _preparar(
        exclusiones=(excluido,), clases={"X": "lumpy"}
    )
    _fijar_de_nuevo(
        conn, rid, protocolo, texto, exclusiones=(excluido,), clases={"X": "lumpy"}
    )
    with pytest.raises(EvidenciaInconsistenteError, match="exclusiones"):
        _fijar_de_nuevo(conn, rid, protocolo, texto)
    assert conn.execute("SELECT * FROM m3_exclusiones").fetchall() == [
        (rid, "X", "lumpy", ELEGIBILIDAD_MENOS_DE_DOS_FAMILIAS, causa)
    ]
    conn.close()


def test_un_sku_evaluado_no_puede_figurar_como_excluido():
    evaluado = SkuExclusion("S1", "historia corta", "eligibility.insufficient_history")
    with pytest.raises(sqlite3.IntegrityError, match="excluido"):
        _preparar(exclusiones=(evaluado,), clases={"S1": "smooth"})


def test_una_exclusion_lleva_un_codigo_de_elegibilidad():
    falla = SkuExclusion("X", "sin ensayos viables", "selection.no_viable_trial")
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        _preparar(exclusiones=(falla,), clases={"X": "lumpy"})
