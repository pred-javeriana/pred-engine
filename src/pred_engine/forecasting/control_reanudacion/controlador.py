"""Coordinacion de M3 con protocolo congelado y recuperacion por fase (3.0)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos.manifiesto_candidatos import ContextoParticion
from pred_engine.forecasting.adaptador_candidatos import (
    FABRICAS,
    FabricaAdaptador,
    LoteRechazadoError,
    validar_manifiesto,
)
from pred_engine.forecasting.evaluaciones.calculo_errores import (
    VERSION_METRICAS,
    EntradaSku,
    SerieCandidato,
    evaluar_sku,
    pronosticar_linea_base,
    separar_reserva,
)
from pred_engine.forecasting.evaluaciones.veredictos import (
    POLITICA_INICIAL,
    PoliticaSeleccion,
    emitir_veredictos,
)
from pred_engine.forecasting.generador_pronosticos import (
    ConfiguracionPronosticos,
    FalloPronostico,
    SerieNoElegibleError,
    generar_pronosticos,
    separar_serie,
    ventanas_reserva,
)
from pred_engine.forecasting.generador_pronosticos.particion import (
    fechas_observadas,
    ventanas_observables,
)
from pred_engine.forecasting.persistencia_evidencia import (
    EvidenciaInconsistenteError,
    IdentidadCorrida,
    contar_fallos,
    estado_corrida,
    guardar_evaluacion,
    guardar_unidades,
    marcar_evaluada,
    publicar,
    registrar_corrida,
    veredictos_publicados,
)
from pred_engine.forecasting.persistencia_evidencia.recuperacion import (
    candidatos_con_fallos,
    cargar_serie,
    cargar_unidades,
    fijar_protocolo,
    guardar_unidad_generada,
    protocolo_guardado,
    restaurar_handoff,
)
from pred_engine.forecasting.persistencia_evidencia.referencias import (
    asegurar_referencias,
    finalizar_referencia,
)
from pred_engine.optimizacion.control_reanudacion.almacenamiento import escribir_atomico
from pred_engine.optimizacion.router import construir_manifiesto

if TYPE_CHECKING:
    from pred_engine.pipeline import EvaluationArtifact, ValidationArtifact

_logger = get_logger(__name__)


def politica_serializable(politica: PoliticaSeleccion) -> dict[str, Any]:
    """Contenido completo de la politica, no solo su etiqueta de version."""
    return {
        "version": politica.version,
        "n_min": politica.n_min,
        "n_min_dm": politica.n_min_dm,
        "alfa_dm": politica.alfa_dm,
        "tolerancia_empate": politica.tolerancia_empate,
        "orden_simplicidad": list(politica.orden_simplicidad),
        "familias_elegibles": {
            c: list(f) for c, f in politica.familias_elegibles.items()
        },
    }


class ControladorEvaluacion:
    """Validador nativo L4: la conexion y el directorio pertenecen al llamador.

    La base de plataforma ya existe y tiene las migraciones M3 aplicadas.
    Todo resultado, tambien un lote rechazado, queda en esa misma base.
    """

    nativo = True

    def __init__(
        self,
        conn: sqlite3.Connection,
        directorio: str | Path,
        *,
        run_id_m2: str,
        codigo_version: str,
        datos_sinteticos: bool | None,
        data_root: str | Path,
        politica: PoliticaSeleccion = POLITICA_INICIAL,
        configuracion_m2: Mapping[str, Any] | None = None,
        fabricas: Mapping[str, FabricaAdaptador] = FABRICAS,
    ) -> None:
        self.conn = conn
        self.directorio = Path(directorio)
        self.data_root = Path(data_root)
        self.run_id_m2 = run_id_m2
        self.codigo_version = codigo_version
        self.datos_sinteticos = datos_sinteticos
        self.politica = politica
        self.configuracion_m2 = dict(configuracion_m2 or {})
        self.fabricas = fabricas

    def __call__(self, evaluation: EvaluationArtifact) -> ValidationArtifact:
        from pred_engine.run_artifacts import sha256_file

        ingesta = evaluation.fitting.ingestion
        ingesta_ref_m1 = sha256_file(ingesta.parquet_path)
        # La posesion pertenece a esta llamada: una colision rechazada no puede
        # finalizar una ejecucion ajena ni heredar la posesion de un intento previo.
        ajustes = evaluation.settings
        ingesta_sha, ejecucion_id, datos_sinteticos = asegurar_referencias(
            self.conn,
            ingesta,
            ingesta_ref_m1=ingesta_ref_m1,
            run_id_m2=self.run_id_m2,
            directorio=self.directorio,
            # Los ajustes evaluados son los que cambian resultados en la base.
            configuracion={
                **self.configuracion_m2,
                "min_train": ajustes.min_train,
                "horizon": ajustes.horizon,
                "step": ajustes.step,
                "seasonality": ajustes.seasonality,
                "metric": ajustes.metric,
            },
            data_root=self.data_root,
            datos_sinteticos=self.datos_sinteticos,
        )
        try:
            resultado = self._evaluar(
                evaluation, ingesta_sha, ejecucion_id, ingesta_ref_m1, datos_sinteticos
            )
        except Exception as error:
            finalizar_referencia(self.conn, ejecucion_id, error=error)
            raise
        finalizar_referencia(
            self.conn,
            ejecucion_id,
            hay_fallos=bool(resultado.audit_bundle["n_fallos"])
            or any(u.failed for u in (*evaluation.fitting.units, *evaluation.units)),
            conservar_completada=resultado.audit_bundle["publicada_previamente"],
        )
        return resultado

    def _evaluar(
        self,
        evaluation: EvaluationArtifact,
        ingesta_sha: str,
        ejecucion_id: int,
        ingesta_ref_m1: str,
        datos_sinteticos: bool | None,
    ) -> ValidationArtifact:
        from pred_engine.run_artifacts import (
            library_versions,
            procedencia_candidato,
        )

        fitting = evaluation.fitting
        panel, corte = fitting.ingestion.panel, fitting.reserve
        contexto = ContextoParticion(
            ingesta_ref_m1=ingesta_ref_m1,
            t_corte_reserva=corte.t_star.date(),
            fraccion_reserva=corte.fraction,
        )
        nuevo = construir_manifiesto(
            [c.selection for c in fitting.candidates],
            run_id_m2=self.run_id_m2,
            contexto=contexto,
            emitido_en=datetime.now(UTC),
        )
        self.directorio.mkdir(parents=True, exist_ok=True)
        ruta_manifiesto = self.directorio / "candidatos.json"
        faltaba_manifiesto = not ruta_manifiesto.exists()
        contenido = nuevo if faltaba_manifiesto else ruta_manifiesto.read_text()
        config = ConfiguracionPronosticos(
            evaluation.settings.horizon, self.codigo_version
        )
        versiones = library_versions()
        protocolo = {
            "versiones": versiones,
            "pronosticos": asdict(config),
            "politica": politica_serializable(self.politica),
            "datos_sinteticos": datos_sinteticos,
            "configuracion_m2": self.configuracion_m2,
        }
        # La identidad de configuracion no incorpora la hora de emision.
        try:
            sobre = json.loads(contenido)
            receta = {k: v for k, v in sobre.items() if k != "emitido_en"}
        except (ValueError, AttributeError):
            receta = {"manifiesto_ilegible": contenido}
        configuracion_id = hashlib.sha256(
            json.dumps(
                {"protocolo": protocolo, "candidatos": receta},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        identidad = IdentidadCorrida(
            ingesta_ref_m1=contexto.ingesta_ref_m1,
            configuracion_id=configuracion_id,
            codigo_version=self.codigo_version,
            t_corte_reserva=contexto.t_corte_reserva,
            version_politica=self.politica.version,
            version_metricas=VERSION_METRICAS,
        )
        apertura = datetime.now(UTC)
        guardado = protocolo_guardado(self.conn, identidad.run_id)
        if guardado is not None:
            contenido = guardado["manifiesto"]
        if faltaba_manifiesto or (
            guardado is not None
            and ruta_manifiesto.read_text().strip() != contenido.strip()
        ):
            escribir_atomico(ruta_manifiesto, contenido + "\n")
        try:
            handoff = restaurar_handoff(
                self.conn, identidad.run_id, manifiesto_pendiente=contenido
            )
            if handoff is None:
                handoff = validar_manifiesto(
                    contenido,
                    esperado=contexto,
                    skus_panel=set(panel["sku_id"]),
                    reserva_abierta_en=apertura,
                )
            elif handoff.contexto != contexto:
                raise LoteRechazadoError(
                    "contexto_incompatible", "particion durable incompatible"
                )
            # Misma entrada y protocolo: una publicacion congelada se reutiliza
            # aunque un reintento de M2 recalcule otro conjunto de candidatos.
            if estado_corrida(self.conn, identidad.run_id) == "PUBLICADA":
                assert guardado is not None
                return self._artefacto(
                    identidad.run_id,
                    config,
                    n_previstas=guardado["n_unidades_previstas"],
                    exclusiones=guardado["procedencia"]["exclusiones_m3"],
                    exclusiones_datos=guardado["procedencia"]["exclusiones_datos"],
                    publicada_previamente=True,
                    datos_sinteticos=datos_sinteticos,
                )
            existente = json.loads(contenido)
            if existente.get("candidatos") != json.loads(nuevo)["candidatos"]:
                raise LoteRechazadoError(
                    "contexto_incompatible",
                    "el conjunto M2 cambio bajo el mismo run_id",
                )
            clases = panel.groupby("sku_id")["sku_class"].first().to_dict()
            if any(c.sku_class != clases[c.sku] for c in handoff.candidatos):
                raise LoteRechazadoError(
                    "contexto_incompatible", "sku_class difiere de M1"
                )
        except LoteRechazadoError as error:
            registrar_corrida(
                self.conn,
                identidad,
                error,
                ingesta_sha256=ingesta_sha,
                datos_sinteticos=datos_sinteticos,
                run_id_m2=self.run_id_m2,
            )
            raise
        run_id = registrar_corrida(
            self.conn,
            identidad,
            handoff,
            ingesta_sha256=ingesta_sha,
            datos_sinteticos=datos_sinteticos,
        )
        plan = ventanas_reserva(corte, config.horizonte)
        skus = sorted(
            {c.sku for c in handoff.candidatos} | {f.sku for f in handoff.fallos}
        )
        exclusiones = {}
        for sku in skus:
            historia = panel.loc[
                (panel["sku_id"] == sku) & (panel["timestamp"] <= corte.t_star)
            ].sort_values("timestamp")
            try:
                separar_serie(
                    sku,
                    pd.Series(
                        historia["demand_qty"].to_numpy(),
                        index=pd.DatetimeIndex(historia["timestamp"]),
                    ),
                    corte,
                    config,
                )
            except SerieNoElegibleError as error:
                exclusiones[sku] = {"codigo": error.codigo, "causa": str(error)}
        elegibles = set(skus) - set(exclusiones)
        series_sku = {sku: _serie(panel, sku) for sku in skus}
        # El mismo calendario finito que usa el generador: un valor no finito
        # es contexto faltante, no una falla del modelo ni de la linea base.
        calendarios = {sku: fechas_observadas(series_sku[sku]) for sku in skus}
        planes = {
            sku: ventanas_observables(corte, config.horizonte, calendarios[sku])
            if sku in elegibles
            else ()
            for sku in skus
        }
        origenes = {
            sku: {v.origen for v in planeadas} for sku, planeadas in planes.items()
        }
        exclusiones_datos = {
            sku: [
                {
                    "origen": v.origen.isoformat(),
                    "causa": "sin_contexto_observado"
                    if v.origen not in origenes[sku]
                    else "sin_datos_objetivo",
                }
                for v in plan
                if v.origen not in origenes[sku]
                or not v.fechas.isin(calendarios[sku]).all()
            ]
            for sku in elegibles
        }
        n_previstas = sum(
            (sum(c.sku == sku for c in handoff.candidatos) + 1) * len(planes[sku])
            for sku in skus
        )
        fijar_protocolo(
            self.conn,
            run_id,
            protocolo=protocolo,
            manifiesto=contenido,
            procedencia={
                "versiones": versiones,
                "candidatos_m2": [
                    procedencia_candidato(crudo["candidato_id"], c.selection)
                    for crudo, c in zip(
                        json.loads(nuevo)["candidatos"], fitting.candidates, strict=True
                    )
                ],
                "run_id_m2": self.run_id_m2,
                "ejecucion_id": ejecucion_id,
                "exclusiones_m2": [asdict(e) for e in fitting.exclusions],
                "exclusiones_m3": exclusiones,
                "exclusiones_datos": exclusiones_datos,
                "fallos_m2": [
                    {"sku": u.sku_id, "familia": u.family, "causa": u.record.error}
                    for u in fitting.units
                    if u.failed
                ],
                "discrepancia_metrica_hpo": evaluation.settings.metric != "rmse",
                "limite": (
                    "seleccion y puntaje sobre la misma reserva; "
                    "evidencia retrospectiva"
                ),
            },
            reserva_abierta_en=apertura,
            n_unidades_previstas=n_previstas,
            ejecucion_id=ejecucion_id,
        )
        # La reserva solo se lee despues del handoff validado y protocolo durable.
        entradas: list[EntradaSku] = []
        bases: dict[str, SerieCandidato] = {}
        if estado_corrida(self.conn, run_id) in ("EN_CURSO", "EVALUADA"):
            fallos_sku = Counter(f.sku for f in handoff.fallos)
            for sku in skus:
                plan_sku = planes[sku]
                serie = series_sku[sku]
                historia, reserva = separar_reserva(serie, corte)
                series = []
                for candidato in (c for c in handoff.candidatos if c.sku == sku):
                    if plan_sku:
                        confirmadas = cargar_unidades(self.conn, run_id, candidato)
                        if estado_corrida(self.conn, run_id) == "EVALUADA" and len(
                            confirmadas
                        ) != len(plan_sku):
                            raise EvidenciaInconsistenteError(
                                "faltan unidades en una corrida EVALUADA"
                            )
                        generar_pronosticos(
                            candidato,
                            serie,
                            corte,
                            config,
                            fabricas=self.fabricas,
                            confirmadas=confirmadas,
                            confirmar=lambda unidad: guardar_unidad_generada(
                                self.conn, run_id, unidad
                            ),
                        )
                    series.append(
                        cargar_serie(
                            self.conn,
                            run_id,
                            sku,
                            candidato.candidato_id,
                            candidato.familia,
                            candidato.modelo,
                        )
                    )
                entrada = EntradaSku(
                    sku,
                    clases[sku],
                    historia,
                    reserva,
                    tuple(series),
                    fallos_sku[sku]
                    + len(candidatos_con_fallos(self.conn, run_id, sku)),
                )
                id_base = f"{sku}/seasonal_naive/seasonal_naive"
                guardada = cargar_serie(
                    self.conn, run_id, sku, id_base, "seasonal_naive", "seasonal_naive"
                )
                if estado_corrida(self.conn, run_id) == "EN_CURSO":
                    base = pronosticar_linea_base(
                        entrada,
                        corte,
                        ventanas_planeadas={v.origen: v.fechas for v in plan_sku},
                        confirmados=guardada.pronosticos,
                    )
                    guardar_unidades(
                        self.conn,
                        run_id,
                        sku,
                        base,
                        codigo_version=self.codigo_version,
                        semilla=0,
                        emitido_en=datetime.now(UTC),
                        fallos={
                            p.origen: FalloPronostico(
                                sku,
                                id_base,
                                "seasonal_naive",
                                p.origen,
                                "valores",
                                "pronostico_invalido",
                                f"sku={sku} linea base sin pronostico valido",
                            )
                            for p in base.pronosticos
                            if not np.isfinite(p.valores).all()
                        },
                    )
                    guardada = cargar_serie(
                        self.conn,
                        run_id,
                        sku,
                        id_base,
                        "seasonal_naive",
                        "seasonal_naive",
                    )
                entradas.append(entrada)
                bases[sku] = guardada
            if estado_corrida(self.conn, run_id) == "EN_CURSO":
                marcar_evaluada(self.conn, run_id)
            evaluaciones = [
                evaluar_sku(e, corte, linea_base=bases[e.sku]) for e in entradas
            ]
            resultado = emitir_veredictos(
                evaluaciones,
                datos_sinteticos=datos_sinteticos,
                politica=self.politica,
            )
            guardar_evaluacion(self.conn, run_id, evaluaciones, resultado)
        return self._artefacto(
            run_id,
            config,
            n_previstas=n_previstas,
            exclusiones=exclusiones,
            exclusiones_datos=exclusiones_datos,
            publicada_previamente=False,
            datos_sinteticos=datos_sinteticos,
        )

    def _artefacto(
        self,
        run_id: str,
        config: ConfiguracionPronosticos,
        *,
        n_previstas: int,
        exclusiones: Mapping[str, Any],
        exclusiones_datos: Mapping[str, Any],
        publicada_previamente: bool,
        datos_sinteticos: bool | None,
    ) -> ValidationArtifact:
        from pred_engine.pipeline import SkuVerdict, ValidationArtifact

        archivos = publicar(self.conn, run_id, self.directorio / "m3")
        filas = veredictos_publicados(self.conn, run_id)
        _logger.info("Evaluacion publicada run_id=%s skus=%d", run_id, len(filas))
        n_fallos = contar_fallos(self.conn, run_id)
        return ValidationArtifact(
            verdicts=tuple(
                SkuVerdict(fila["sku"], fila["veredicto"], fila) for fila in filas
            ),
            audit_bundle={
                "run_id_m3": run_id,
                "estado": "PUBLICADA",
                "datos_sinteticos": datos_sinteticos,
                "horizonte": config.horizonte,
                "n_unidades": n_previstas,
                "n_fallos": n_fallos,
                "exclusiones": exclusiones,
                "exclusiones_datos": exclusiones_datos,
                "publicada_previamente": publicada_previamente,
                "archivos": {tabla: str(path) for tabla, path in archivos.items()},
                "csv": {
                    tabla: str(path.with_suffix(".csv"))
                    for tabla, path in archivos.items()
                },
            },
        )


def _serie(panel: pd.DataFrame, sku: str) -> pd.Series:
    grupo = panel.loc[panel["sku_id"] == sku].sort_values("timestamp")
    return pd.Series(
        grupo["demand_qty"].to_numpy(), index=pd.DatetimeIndex(grupo["timestamp"])
    )
