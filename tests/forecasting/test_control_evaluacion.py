"""M2 real, evaluacion reservada nativa y recuperacion transaccional M3."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests._plataforma import crear_base_plataforma

from pred_engine.forecasting.adaptador_candidatos import LoteRechazadoError
from pred_engine.forecasting.base import SeasonalNaiveStub
from pred_engine.forecasting.control_reanudacion import ControladorEvaluacion
from pred_engine.forecasting.persistencia_evidencia import aplicar_migraciones_m3
from pred_engine.optimizacion.optimizadores.modelos_clasicos import fabrica_sarima
from pred_engine.optimizacion.router import (
    SelectionResult,
    SelectionRouter,
    StrategyRegistry,
)
from pred_engine.pipeline import EvaluationSettings, PipelineInput
from pred_engine.pipeline_setup import ConfiguredTopologyPolicy
from pred_engine.verify import representative_csv, verification_pipeline

pytestmark = pytest.mark.slow


def _caso(tmp_path, conn, **opciones):
    csv = representative_csv(tmp_path / "fuente.csv")
    pipeline = verification_pipeline()
    pipeline.validator = ControladorEvaluacion(
        conn,
        tmp_path / "corrida",
        run_id_m2="m2-integral",
        codigo_version="revision-1",
        datos_sinteticos=True,
        data_root=tmp_path / "data",
        **opciones,
    )
    return pipeline, PipelineInput(csv_path=csv, data_root=tmp_path / "data")


def _conexion(tmp_path, *, factory=sqlite3.Connection):
    path = crear_base_plataforma(tmp_path / "central.db")
    conn = sqlite3.connect(path, factory=factory)
    aplicar_migraciones_m3(conn)
    return conn


def test_m2_a_m3_publica_todas_las_ventanas_y_los_dos_skus(tmp_path):
    conn = _conexion(tmp_path)
    pipeline, request = _caso(tmp_path, conn)
    datos = pd.read_csv(request.csv_path)
    pd.concat(
        [datos, datos.assign(sku_id="segundo", demand_qty=datos.demand_qty + 100)]
    ).to_csv(request.csv_path, index=False)
    resultado = pipeline.run(request)
    assert resultado.complete is True
    assert resultado.blocked_stage is None
    assert [(v.sku_id, v.verdict) for v in resultado.validation.verdicts] == [
        ("proof-sku", "EVIDENCIA_INSUFICIENTE"),
        ("segundo", "EVIDENCIA_INSUFICIENTE"),
    ]
    assert conn.execute("SELECT estado FROM m3_corridas").fetchall() == [("PUBLICADA",)]
    assert conn.execute("SELECT COUNT(*) FROM m3_unidades").fetchone()[0] == 24
    assert conn.execute("SELECT COUNT(*) FROM m3_pronosticos").fetchone()[0] == 48
    base = conn.execute(
        "SELECT valor FROM m3_pronosticos "
        "WHERE candidato_id = 'proof-sku/seasonal_naive/seasonal_naive' "
        "AND origen = '2024-01-28' ORDER BY h"
    ).fetchall()
    assert [fila[0] for fila in base] == pytest.approx([26.35, 27.7])
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert conn.execute("SELECT ejecucion_id FROM m3_protocolos").fetchone()[0] == 1
    protocolo, procedencia = (
        json.loads(texto)
        for texto in conn.execute(
            "SELECT protocolo, procedencia FROM m3_protocolos"
        ).fetchone()
    )
    # Politica y bloques quedan fijos con el protocolo, antes de abrir la reserva:
    # reserva de 7 dias con H = 2, seleccion de 2 dias y 4 origenes de prueba.
    politica = protocolo["politica"]
    assert (politica["version"], politica["fraccion_seleccion"]) == (
        "3.4.4-bloques",
        0.4,
    )
    assert (
        politica["n_min_confirmacion"],
        politica["bloque_bootstrap"],
        politica["replicas_bootstrap"],
        politica["semilla_bootstrap"],
    ) == (10, 14, 2000, 20261029)
    bloques = protocolo["bloques"]
    assert bloques["dias_seleccion"] == 2
    assert (bloques["origenes_seleccion"]["n"], bloques["origenes_prueba"]) == (
        1,
        {"n": 4, "primero": "2024-01-30", "ultimo": "2024-02-02"},
    )
    assert procedencia["limite"] == (
        "seleccion y puntaje en bloques disjuntos de la misma reserva; "
        "evidencia retrospectiva"
    )
    assert [s["sku_class"] for s in procedencia["seleccion_previa"]] == ["smooth"]
    assert conn.execute(
        "SELECT n_skus, n_instancias, n_calculables FROM m3_cartera "
        "WHERE familia = 'classical' AND metrica = 'rmse'"
    ).fetchall() == [(2, 2, 2)]
    exports = resultado.validation.audit_bundle["archivos"]
    assert pd.read_csv(Path(exports["m3_veredictos"]).with_suffix(".csv"))[
        "sku"
    ].tolist() == ["proof-sku", "segundo"]
    assert pd.read_parquet(exports["m3_pronosticos"]).shape[0] == 48
    conn.close()


@dataclass(frozen=True)
class _SarimaQueAnotaProceso:
    """Fabrica serializable: deja el pid del proceso que ajusta el modelo."""

    carpeta: Path

    def __call__(self, configuracion, *, seed):
        (self.carpeta / str(os.getpid())).touch()
        return fabrica_sarima(configuracion, seed=seed)


def _evidencia_m3(conn):
    """Toda la evidencia M3 salvo las horas de emision, que cambian por corrida."""
    unidades = conn.execute("SELECT * FROM m3_unidades ORDER BY rowid")
    columnas = [c[0] for c in unidades.description]
    return {
        "unidades": [
            {k: v for k, v in zip(columnas, fila, strict=True) if k != "emitido_en"}
            for fila in unidades.fetchall()
        ],
        **{
            tabla: conn.execute(f"SELECT * FROM {tabla} ORDER BY rowid").fetchall()
            for tabla in ("m3_pronosticos", "m3_metricas", "m3_veredictos")
        },
    }


def test_l4_en_paralelo_publica_lo_mismo_que_en_un_proceso(tmp_path):
    procesos_ajuste = tmp_path / "pids"
    procesos_ajuste.mkdir()
    inicial = _conexion(tmp_path / "inicial")
    pipeline, request = _caso(tmp_path, inicial)
    datos = pd.read_csv(request.csv_path)
    pd.concat(
        [datos, datos.assign(sku_id="segundo", demand_qty=datos.demand_qty + 100)]
    ).to_csv(request.csv_path, index=False)
    evaluation = pipeline.evaluate(pipeline.fit(pipeline.ingest(request)))
    inicial.close()
    evidencias = []
    for procesos in (1, 2):
        conn = _conexion(tmp_path / str(procesos))
        pipeline.validator = ControladorEvaluacion(
            conn,
            tmp_path / f"corrida-{procesos}",
            run_id_m2="m2-integral",
            codigo_version="revision-1",
            datos_sinteticos=True,
            data_root=tmp_path / "data",
            fabricas={"classical": _SarimaQueAnotaProceso(procesos_ajuste)},
            procesos=procesos,
        )
        resultado = pipeline.validate(evaluation)
        assert resultado.audit_bundle["estado"] == "PUBLICADA"
        evidencias.append(_evidencia_m3(conn))
        conn.close()
    assert len(evidencias[0]["unidades"]) == 24
    assert evidencias[1] == evidencias[0]
    pids = {int(p.name) for p in procesos_ajuste.iterdir()}
    assert os.getpid() in pids
    assert len(pids - {os.getpid()}) >= 1


def test_publicada_se_relee_sin_modelos_ni_reescritura(tmp_path, monkeypatch):
    conn = _conexion(tmp_path)
    pipeline, request = _caso(tmp_path, conn)
    fitting = pipeline.fit(pipeline.ingest(request))
    evaluation = pipeline.evaluate(fitting)
    primera = pipeline.validate(evaluation)
    unidades = conn.execute(
        "SELECT * FROM m3_unidades ORDER BY origen, candidato_id"
    ).fetchall()
    archivos = {
        p: Path(p).read_bytes() for p in primera.audit_bundle["archivos"].values()
    }

    def imposible(configuracion, *, seed):
        raise KeyboardInterrupt("no se debe ajustar una corrida publicada")

    pipeline.validator.fabricas = {"classical": imposible}
    emision = json.loads((tmp_path / "corrida" / "candidatos.json").read_text())[
        "emitido_en"
    ]
    (tmp_path / "corrida" / "candidatos.json").unlink()

    def no_revalidar(*args, **kwargs):
        raise KeyboardInterrupt("una corrida durable no vuelve a cargar pesos")

    monkeypatch.setattr(
        "pred_engine.forecasting.control_reanudacion.controlador.validar_manifiesto",
        no_revalidar,
    )
    segunda = pipeline.validate(evaluation)
    assert (
        json.loads((tmp_path / "corrida" / "candidatos.json").read_text())["emitido_en"]
        == emision
    )
    assert [(v.sku_id, v.verdict) for v in segunda.verdicts] == [
        ("proof-sku", "EVIDENCIA_INSUFICIENTE")
    ]
    assert segunda.audit_bundle["run_id_m3"] == primera.audit_bundle["run_id_m3"]
    assert (
        conn.execute(
            "SELECT * FROM m3_unidades ORDER BY origen, candidato_id"
        ).fetchall()
        == unidades
    )
    assert {p: Path(p).read_bytes() for p in archivos} == archivos
    conn.close()


@pytest.mark.parametrize("procesos", [1, 2])
@pytest.mark.parametrize("despues", [False, True])
def test_caida_antes_y_despues_del_commit_preserva_la_unidad_durable(
    tmp_path, despues, procesos
):
    class ConexionConCaida(sqlite3.Connection):
        armada = False

        def commit(self):
            if (
                self.armada
                and self.execute("SELECT COUNT(*) FROM m3_unidades").fetchone()[0]
            ):
                self.armada = False
                if despues:
                    super().commit()
                raise KeyboardInterrupt("caida de proceso")
            super().commit()

    conn = _conexion(tmp_path, factory=ConexionConCaida)
    pipeline, request = _caso(tmp_path, conn, procesos=procesos)
    evaluation = pipeline.evaluate(pipeline.fit(pipeline.ingest(request)))
    conn.armada = True
    with pytest.raises(KeyboardInterrupt, match="caida de proceso"):
        pipeline.validate(evaluation)
    # Las ventanas nuevas de los candidatos de un SKU van en un solo commit: el
    # SARIMA de proof-sku tiene 6 y quedan todas durables o ninguna.
    assert conn.execute("SELECT COUNT(*) FROM m3_unidades").fetchone()[0] == (
        6 if despues else 0
    )
    assert conn.execute("SELECT estado FROM m3_corridas").fetchone()[0] == "EN_CURSO"
    previa = conn.execute(
        "SELECT origen, emitido_en FROM m3_unidades ORDER BY origen"
    ).fetchall()
    final = pipeline.validate(evaluation)
    assert final.audit_bundle["estado"] == "PUBLICADA"
    assert conn.execute("SELECT COUNT(*) FROM m3_unidades").fetchone()[0] == 12
    assert conn.execute("SELECT COUNT(*) FROM m3_pronosticos").fetchone()[0] == 24
    if despues:
        assert (
            conn.execute(
                "SELECT origen, emitido_en FROM m3_unidades "
                "WHERE candidato_id = 'proof-sku/classical/sarima' ORDER BY origen"
            ).fetchall()
            == previa
        )
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()


@pytest.mark.parametrize("fase", ["EVALUADA", "SELECCIONADA"])
def test_reanuda_fases_durables_sin_ajustar_ni_cambiar_unidades(
    tmp_path, monkeypatch, fase
):
    class ConexionConCaida(sqlite3.Connection):
        armada = False

        def commit(self):
            fila = (
                self.execute("SELECT estado FROM m3_corridas").fetchone()
                if self.armada
                else None
            )
            super().commit()
            if self.armada and fila == (fase,):
                self.armada = False
                raise KeyboardInterrupt("caida despues de fase durable")

    conn = _conexion(tmp_path, factory=ConexionConCaida)
    pipeline, request = _caso(tmp_path, conn)
    evaluation = pipeline.evaluate(pipeline.fit(pipeline.ingest(request)))
    conn.armada = True
    with pytest.raises(KeyboardInterrupt, match="fase durable"):
        pipeline.validate(evaluation)
    assert conn.execute("SELECT estado FROM m3_corridas").fetchone()[0] == fase
    unidades = conn.execute(
        "SELECT * FROM m3_unidades ORDER BY candidato_id, origen"
    ).fetchall()
    assert len(unidades) == 12

    def no_ajustar(*args, **kwargs):
        raise KeyboardInterrupt("no se ajusta una fase con ventanas completas")

    pipeline.validator.fabricas = {"classical": no_ajustar}
    # Seasonal Naive tampoco se ajusta en los pliegues L3: la seleccion de
    # robustez se lee de la procedencia durable del protocolo.
    monkeypatch.setattr(SeasonalNaiveStub, "fit", no_ajustar)
    resultado = pipeline.validate(evaluation)
    assert resultado.audit_bundle["estado"] == "PUBLICADA"
    assert (
        conn.execute(
            "SELECT * FROM m3_unidades ORDER BY candidato_id, origen"
        ).fetchall()
        == unidades
    )
    assert conn.execute("SELECT COUNT(*) FROM m3_seleccion_previa").fetchone()[0] > 0
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()


def test_caida_despues_del_registro_reusa_la_validacion_durable(tmp_path, monkeypatch):
    class ConexionConCaida(sqlite3.Connection):
        armada = False

        def commit(self):
            if not self.armada:
                return super().commit()
            primera = self.execute("SELECT COUNT(*) FROM m3_corridas").fetchone()[0]
            fijadas = self.execute("SELECT COUNT(*) FROM m3_protocolos").fetchone()[0]
            super().commit()
            if self.armada and primera and not fijadas:
                self.armada = False
                raise KeyboardInterrupt("caida despues de registrar")

    conn = _conexion(tmp_path, factory=ConexionConCaida)
    pipeline, request = _caso(tmp_path, conn)
    evaluation = pipeline.evaluate(pipeline.fit(pipeline.ingest(request)))
    conn.armada = True
    with pytest.raises(KeyboardInterrupt, match="registrar"):
        pipeline.validate(evaluation)
    assert conn.execute("SELECT COUNT(*) FROM m3_unidades").fetchone()[0] == 0

    def no_revalidar(*args, **kwargs):
        raise KeyboardInterrupt("no debe cambiar el resultado durable de 3.2")

    monkeypatch.setattr(
        "pred_engine.forecasting.control_reanudacion.controlador.validar_manifiesto",
        no_revalidar,
    )
    resultado = pipeline.validate(evaluation)
    assert resultado.audit_bundle["estado"] == "PUBLICADA"
    assert conn.execute("SELECT COUNT(*) FROM m3_corridas").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM m3_unidades").fetchone()[0] == 12
    conn.close()


def test_fallo_de_un_sku_es_visible_y_no_imputa_ceros_ni_detiene_otro(tmp_path):
    class FalloDeUnaSerie(SeasonalNaiveStub):
        def fit(self, y):
            if y[0] < 50:
                raise ValueError("fallo aislado de ajuste")
            return super().fit(y)

    conn = _conexion(tmp_path)
    pipeline, request = _caso(
        tmp_path,
        conn,
        fabricas={"classical": lambda config, *, seed: FalloDeUnaSerie(1, seed)},
    )
    datos = pd.read_csv(request.csv_path)
    pd.concat(
        [datos, datos.assign(sku_id="segundo", demand_qty=datos.demand_qty + 100)]
    ).to_csv(request.csv_path, index=False)
    resultado = pipeline.run(request)
    assert [(v.sku_id, v.verdict) for v in resultado.validation.verdicts] == [
        ("proof-sku", "FALLO_TECNICO"),
        ("segundo", "EVIDENCIA_INSUFICIENTE"),
    ]
    assert resultado.validation.audit_bundle["n_fallos"] == 6
    assert conn.execute(
        "SELECT DISTINCT estado FROM m3_unidades "
        "WHERE candidato_id = 'proof-sku/classical/sarima'"
    ).fetchall() == [("fallida",)]
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM m3_pronosticos "
            "WHERE candidato_id = 'proof-sku/classical/sarima' AND valor IS NULL"
        ).fetchone()[0]
        == 12
    )
    causa = json.loads(
        conn.execute(
            "SELECT causa FROM m3_unidades WHERE estado = 'fallida' LIMIT 1"
        ).fetchone()[0]
    )
    assert (causa["sku"], causa["codigo_error"], causa["campo"]) == (
        "proof-sku",
        "ajuste_fallido",
        "fit",
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM m3_pronosticos "
            "WHERE candidato_id = 'segundo/classical/sarima' AND valor IS NOT NULL"
        ).fetchone()[0]
        == 12
    )
    conn.close()


@pytest.mark.parametrize("con_sku_sano", [False, True])
def test_todos_los_ajustes_l2_fallidos_se_publican_con_causa_y_categoria(
    tmp_path, con_sku_sano
):
    class FalloL2(SeasonalNaiveStub):
        def fit(self, y):
            if y[0] < 50:
                raise ValueError("causa concreta del ajuste L2")
            return super().fit(y)

    conn = _conexion(tmp_path)
    pipeline, request = _caso(tmp_path, conn)
    pipeline.factories = {"classical": lambda config, *, seed: FalloL2(1, seed)}
    if con_sku_sano:
        datos = pd.read_csv(request.csv_path)
        pd.concat(
            [datos, datos.assign(sku_id="segundo", demand_qty=datos.demand_qty + 100)]
        ).to_csv(request.csv_path, index=False)
    resultado = pipeline.run(request)
    assert resultado.complete is True
    esperados = [("proof-sku", "FALLO_TECNICO")]
    if con_sku_sano:
        esperados.append(("segundo", "EVIDENCIA_INSUFICIENTE"))
    assert [(v.sku_id, v.verdict) for v in resultado.validation.verdicts] == esperados
    assert (
        "causa concreta del ajuste L2"
        in resultado.validation.verdicts[0].evidence["justificacion"]
    )
    categorias = conn.execute(
        "SELECT conteos, porcentajes FROM m3_categorias"
    ).fetchall()
    assert len(categorias) == 1
    conteos, porcentajes = map(json.loads, categorias[0])
    assert conteos["FALLO_TECNICO"] == 1
    assert porcentajes["FALLO_TECNICO"] == (50.0 if con_sku_sano else 100.0)
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM m3_unidades WHERE sku = 'proof-sku'"
        ).fetchone()[0]
        == 6
    )
    assert resultado.validation.audit_bundle["n_unidades"] == (
        18 if con_sku_sano else 6
    )
    assert conn.execute("SELECT estado FROM m3_corridas").fetchall() == [("PUBLICADA",)]
    conn.close()


def test_candidato_l2_rechazado_por_3_2_conserva_su_veredicto_ante_otro_fallo_l2(
    tmp_path,
):
    @dataclass(frozen=True)
    class SeleccionSinEstacional:
        family: str

        def select(self, request, profile):
            # Ajusta en L2, pero 3.2 exige todas las claves de la familia.
            return SelectionResult(
                sku_id=request.sku_id,
                sku_class=request.sku_class,
                family=self.family,
                profile=profile,
                produced_by="seleccion-sin-estacional",
                forecast_config={"p": 0, "d": 0, "q": 0},
            )

    class SeleccionFallida:
        family = "dl"

        def select(self, request, profile):
            raise ValueError("causa concreta de la otra familia L2")

    def veredicto(raiz, familias):
        conn = _conexion(raiz)
        pipeline, request = _caso(raiz, conn)
        registry = StrategyRegistry()
        registry.register("classical", SeleccionSinEstacional("classical"))
        registry.register("ml", SeleccionSinEstacional("ml"))
        registry.register("dl", SeleccionFallida())
        pipeline.router = SelectionRouter(ConfiguredTopologyPolicy(familias), registry)

        def fabrica(config, *, seed):
            return SeasonalNaiveStub(1, seed)

        pipeline.factories = {"classical": fabrica, "ml": fabrica, "dl": fabrica}
        resultado = pipeline.run(request)
        conn.close()
        assert [u.failed for u in resultado.fitting.units] == [
            familia == "dl" for familia in familias
        ]
        (unico,) = resultado.validation.verdicts
        return {k: v for k, v in unico.evidence.items() if k != "run_id"}

    # Dos familias entregadas en L2: la regla de dos familias no excluye el SKU.
    solo_rechazo = veredicto(tmp_path / "sin-fallo-l2", ("classical", "ml"))
    con_fallo_l2 = veredicto(tmp_path / "con-fallo-l2", ("classical", "ml", "dl"))
    assert solo_rechazo["veredicto"] == "FALLO_TECNICO"
    assert con_fallo_l2 == solo_rechazo


def test_fallo_parcial_marca_comparacion_incompleta_y_conserva_ventanas(tmp_path):
    class FalloInicial(SeasonalNaiveStub):
        def pronosticar(self, observadas, horizon):
            if not len(observadas):
                return np.full(horizon, np.nan)
            return super().pronosticar(observadas, horizon)

    conn = _conexion(tmp_path)
    pipeline, request = _caso(
        tmp_path,
        conn,
        fabricas={"classical": lambda config, *, seed: FalloInicial(1, seed)},
    )
    resultado = pipeline.run(request)
    evidencia = resultado.validation.verdicts[0].evidence
    assert resultado.validation.audit_bundle["n_fallos"] == 1
    assert evidencia["comparacion_incompleta"] == 1
    assert conn.execute("SELECT COUNT(*) FROM m3_unidades").fetchone()[0] == 12
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM m3_pronosticos WHERE valor IS NULL"
        ).fetchone()[0]
        == 2
    )
    conn.close()


def test_parquet_de_m1_entra_sin_transformacion_ni_republicacion(tmp_path):
    conn = _conexion(tmp_path)
    pipeline, request = _caso(tmp_path, conn)
    ingesta = pipeline.ingest(request)
    contenido = ingesta.parquet_path.read_bytes()
    assert ingesta.panel.columns.tolist() == [
        "sku_id",
        "timestamp",
        "demand_qty",
        "lead_time_days",
        "sku_class",
    ]
    resultado = pipeline.run(
        PipelineInput(
            parquet_path=ingesta.parquet_path,
            data_root=tmp_path / "sin_publicacion",
        )
    )
    assert resultado.complete is True
    # Reserva de 7 dias con H = 2: el bloque de prueba tiene los origenes 2 a 5.
    assert resultado.validation.verdicts[0].evidence["n_ventanas"] == 4
    assert ingesta.parquet_path.read_bytes() == contenido
    assert not (tmp_path / "sin_publicacion").exists()
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()


def test_handoff_incompatible_queda_rechazado_sin_abrir_ventanas(tmp_path):
    conn = _conexion(tmp_path)
    pipeline, request = _caso(tmp_path, conn)
    evaluation = pipeline.evaluate(pipeline.fit(pipeline.ingest(request)))
    carpeta = tmp_path / "corrida"
    carpeta.mkdir()
    (carpeta / "candidatos.json").write_text('{"schema_version": 99}')
    with pytest.raises(LoteRechazadoError) as error:
        pipeline.validate(evaluation)
    assert error.value.motivo == "version_no_soportada"
    assert conn.execute("SELECT estado, run_id_m2 FROM m3_corridas").fetchall() == [
        ("RECHAZADA", "m2-integral")
    ]
    assert conn.execute("SELECT COUNT(*) FROM m3_unidades").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM evidencia_publicada").fetchone()[0] == 0
    conn.close()


def test_historia_corta_entregada_por_m2_se_excluye_antes_de_ajustar(tmp_path):
    class SeleccionFija:
        family = "classical"

        def select(self, request, profile):
            return SelectionResult(
                sku_id=request.sku_id,
                sku_class=request.sku_class,
                family="classical",
                profile=profile,
                produced_by="seleccion-fija",
                forecast_config={
                    "p": 0,
                    "d": 0,
                    "q": 0,
                    "P": 0,
                    "D": 0,
                    "Q": 0,
                    "m": 0,
                    "tendencia": "c",
                },
            )

    conn = _conexion(tmp_path)
    pipeline, request = _caso(tmp_path, conn)
    pd.read_csv(request.csv_path).head(8).to_csv(request.csv_path, index=False)
    registry = StrategyRegistry()
    registry.register("classical", SeleccionFija())
    pipeline.router = SelectionRouter(
        ConfiguredTopologyPolicy(("classical",)), registry
    )
    pipeline.factories = {
        "classical": lambda config, *, seed: SeasonalNaiveStub(1, seed)
    }
    pipeline.settings = EvaluationSettings(min_train=2, horizon=1, step=1)
    resultado = pipeline.run(request)
    assert resultado.complete is True
    assert resultado.validation.verdicts[0].verdict == "EVIDENCIA_INSUFICIENTE"
    assert (
        resultado.validation.audit_bundle["exclusiones"]["proof-sku"]["codigo"]
        == "INSUFFICIENT_HISTORY"
    )
    assert conn.execute("SELECT COUNT(*) FROM m3_unidades").fetchone()[0] == 0
    conn.close()


def test_horizonte_mayor_que_reserva_es_insuficiente_sin_truncarlo(tmp_path):
    conn = _conexion(tmp_path)
    pipeline, request = _caso(tmp_path, conn)
    evaluation = pipeline.evaluate(pipeline.fit(pipeline.ingest(request)))
    evaluation = replace(evaluation, settings=replace(evaluation.settings, horizon=8))
    resultado = pipeline.validate(evaluation)
    assert [(v.sku_id, v.verdict) for v in resultado.verdicts] == [
        ("proof-sku", "EVIDENCIA_INSUFICIENTE")
    ]
    assert resultado.audit_bundle["horizonte"] == 8
    assert resultado.audit_bundle["n_unidades"] == 0
    assert conn.execute("SELECT COUNT(*) FROM m3_pronosticos").fetchone()[0] == 0
    conn.close()


def _datos_m4(conn):
    return {
        tabla: conn.execute(f"SELECT * FROM {tabla} ORDER BY rowid").fetchall()
        for tabla in ("m3_cortes", "m3_lanzamientos_tardios", "m3_observaciones")
    }


@pytest.mark.parametrize("fase", ["EN_CURSO", "EVALUADA", "SELECCIONADA"])
@pytest.mark.parametrize("anterior_a_m4", [False, True])
def test_reanudar_conserva_el_corte_y_la_demanda_observada_sin_duplicar(
    tmp_path, fase, anterior_a_m4
):
    """Caida tras fijar el protocolo o tras una fase durable, y reanudacion.

    ``anterior_a_m4`` simula una corrida que llego a esa fase con un motor sin
    la migracion 10: al reanudar recibe su corte si todavia no se evaluo; si
    ya paso de EVALUADA, queda sin corte ni demanda observada (metadata_missing).
    """

    class ConexionConCaida(sqlite3.Connection):
        armada = False

        def commit(self):
            if not self.armada:
                return super().commit()
            estado = self.execute("SELECT estado FROM m3_corridas").fetchone()
            fijado = self.execute("SELECT COUNT(*) FROM m3_protocolos").fetchone()[0]
            super().commit()
            if estado == (fase,) and fijado:
                self.armada = False
                raise KeyboardInterrupt("caida despues de fase durable")

    conn = _conexion(tmp_path, factory=ConexionConCaida)
    pipeline, request = _caso(tmp_path, conn)
    evaluation = pipeline.evaluate(pipeline.fit(pipeline.ingest(request)))
    conn.armada = True
    with pytest.raises(KeyboardInterrupt, match="fase durable"):
        pipeline.validate(evaluation)
    assert conn.execute("SELECT estado FROM m3_corridas").fetchone()[0] == fase
    # El corte va en la misma transaccion que el protocolo.
    t_star = evaluation.fitting.reserve.t_star.strftime("%Y-%m-%d")
    assert conn.execute("SELECT t_star FROM m3_cortes").fetchall() == [(t_star,)]
    if anterior_a_m4:
        for tabla in ("m3_observaciones", "m3_lanzamientos_tardios", "m3_cortes"):
            conn.execute(f"DELETE FROM {tabla}")
        conn.commit()

    resultado = pipeline.validate(evaluation)

    assert resultado.audit_bundle["estado"] == "PUBLICADA"
    datos = _datos_m4(conn)
    if anterior_a_m4 and fase == "SELECCIONADA":
        assert datos == {
            "m3_cortes": [],
            "m3_lanzamientos_tardios": [],
            "m3_observaciones": [],
        }
    else:
        assert [f[1] for f in datos["m3_cortes"]] == [t_star]
        reservados = evaluation.fitting.reserve.reserved_days
        assert len(datos["m3_observaciones"]) == reservados
        assert (
            conn.execute(
                "SELECT sku, fecha FROM m3_pronosticos EXCEPT "
                "SELECT sku, fecha FROM m3_observaciones"
            ).fetchall()
            == []
        )
    # Releer la corrida publicada no cambia nada.
    pipeline.validate(evaluation)
    assert _datos_m4(conn) == datos
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()
