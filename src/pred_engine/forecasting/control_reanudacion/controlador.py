"""Coordinacion de M3 con protocolo congelado y recuperacion por fase (3.0)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pred_engine.comun.ejecucion_paralela import (
    EtiquetaUnidad,
    emitir_evento,
    iterar_unidades,
)
from pred_engine.comun.exclusiones import SkuExclusion
from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos import SkuClass
from pred_engine.comun.modelos.manifiesto_candidatos import (
    Candidato,
    ContextoParticion,
)
from pred_engine.comun.reserva import ReserveCut
from pred_engine.forecasting.adaptador_candidatos import (
    FABRICAS,
    FabricaAdaptador,
    HandoffValidado,
    LoteRechazadoError,
    validar_manifiesto,
)
from pred_engine.forecasting.evaluaciones.calculo_errores import (
    FAMILIA_LINEA_BASE,
    MODELO_LINEA_BASE,
    VERSION_METRICAS,
    BloquesReserva,
    EntradaSku,
    EvaluacionSku,
    PronosticoFechado,
    SerieCandidato,
    evaluar_bloques,
    pronosticar_linea_base,
    separar_reserva,
)
from pred_engine.forecasting.evaluaciones.veredictos import (
    POLITICA_INICIAL,
    CandidatoPrevio,
    PoliticaSeleccion,
    SeleccionCategoria,
    emitir_veredictos,
    evaluacion_previa,
    seleccionar_previa,
)
from pred_engine.forecasting.generador_pronosticos import (
    ConfiguracionPronosticos,
    FalloPronostico,
    SerieNoElegibleError,
    UnidadPronostico,
    VentanaReserva,
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
    guardar_unidades_generadas,
    protocolo_de_ejecucion,
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
        "fraccion_seleccion": politica.fraccion_seleccion,
        "n_min_confirmacion": politica.n_min_confirmacion,
        "bloque_bootstrap": politica.bloque_bootstrap,
        "replicas_bootstrap": politica.replicas_bootstrap,
        "semilla_bootstrap": politica.semilla_bootstrap,
    }


class ControladorEvaluacion:
    """Validador nativo L4: la conexion y el directorio pertenecen al llamador.

    La base de plataforma ya existe y tiene las migraciones M3 aplicadas.
    Todo resultado, tambien un lote rechazado, queda en esa misma base.
    ``procesos`` es la cantidad de procesos que pronostican los SKU; con 1
    todo corre en el proceso actual. Solo este proceso escribe en la base.
    Con ``progreso`` (ruta de ``progreso.jsonl``) anota el plan de L4 y el fin
    de cada candidato, con el formato de L2 y L3, una vez guardado su SKU.
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
        procesos: int = 1,
        progreso: str | Path | None = None,
    ) -> None:
        if type(procesos) is not int or procesos < 1:
            raise ValueError("procesos debe ser un entero positivo")
        self.conn = conn
        self.directorio = Path(directorio)
        self.data_root = Path(data_root)
        self.run_id_m2 = run_id_m2
        self.codigo_version = codigo_version
        self.datos_sinteticos = datos_sinteticos
        self.politica = politica
        self.configuracion_m2 = dict(configuracion_m2 or {})
        self.fabricas = fabricas
        self.procesos = procesos
        self.progreso = progreso

    def __call__(self, evaluation: EvaluationArtifact) -> ValidationArtifact:
        ingesta = evaluation.fitting.ingestion
        # La huella de los bytes que L1 leyo; el archivo actual puede ser otro.
        if ingesta.parquet_sha256 is None:
            raise ValueError(
                "la ingestion no declara parquet_sha256: use Pipeline.ingest o "
                "ingestion_artifact_from_publication para conservar la huella de "
                "los bytes leidos"
            )
        ingesta_ref_m1 = ingesta.parquet_sha256
        configuracion = self._configuracion_efectiva(evaluation)
        # La posesion pertenece a esta llamada: una colision rechazada no puede
        # finalizar una ejecucion ajena ni heredar la posesion de un intento previo.
        ingesta_sha, ejecucion_id, datos_sinteticos = asegurar_referencias(
            self.conn,
            ingesta,
            ingesta_ref_m1=ingesta_ref_m1,
            run_id_m2=self.run_id_m2,
            directorio=self.directorio,
            configuracion=configuracion,
            data_root=self.data_root,
            datos_sinteticos=self.datos_sinteticos,
        )
        try:
            resultado = self._evaluar(
                evaluation,
                ingesta_sha,
                ejecucion_id,
                ingesta_ref_m1,
                datos_sinteticos,
                configuracion,
            )
        except Exception as error:
            finalizar_referencia(
                self.conn, ejecucion_id, error=error, conservar_completada=True
            )
            raise
        finalizar_referencia(
            self.conn,
            ejecucion_id,
            hay_fallos=bool(resultado.audit_bundle["n_fallos"])
            or any(u.failed for u in (*evaluation.fitting.units, *evaluation.units)),
            conservar_completada=resultado.audit_bundle["publicada_previamente"],
        )
        return resultado

    def _configuracion_efectiva(self, evaluation: EvaluationArtifact) -> dict[str, Any]:
        """Los ajustes de M2 y de la evaluacion que produjeron los resultados.

        La configuracion del llamador puede agregar campos, pero no contradecir
        un ajuste efectivo: un conflicto se rechaza antes de registrar nada.
        """
        ajustes = evaluation.settings
        efectivos = {
            **evaluation.fitting.m2_settings,
            "code": self.codigo_version,
            "min_train": ajustes.min_train,
            "horizon": ajustes.horizon,
            "step": ajustes.step,
            "seasonality": ajustes.seasonality,
            "metric": ajustes.metric,
        }
        distintos = sorted(
            k
            for k, v in efectivos.items()
            if k in self.configuracion_m2
            and json.loads(json.dumps(self.configuracion_m2[k]))
            != json.loads(json.dumps(v))
        )
        if distintos:
            raise ValueError(
                "la configuracion M2 declarada difiere de la efectiva: "
                + ", ".join(distintos)
            )
        return {**self.configuracion_m2, **efectivos}

    def _evaluar(
        self,
        evaluation: EvaluationArtifact,
        ingesta_sha: str,
        ejecucion_id: int,
        ingesta_ref_m1: str,
        datos_sinteticos: bool | None,
        configuracion_m2: Mapping[str, Any],
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
        # Los bloques quedan fijos con el protocolo, antes de abrir la reserva.
        bloques = BloquesReserva.de(
            corte, config.horizonte, self.politica.fraccion_seleccion
        )
        versiones = library_versions()
        protocolo = {
            "versiones": versiones,
            "pronosticos": asdict(config),
            "politica": politica_serializable(self.politica),
            "bloques": bloques.limites(),
            "datos_sinteticos": datos_sinteticos,
            "configuracion_m2": configuracion_m2,
        }
        # SQLite manda sobre archivos locales y ajustes finales transitorios.
        previo = protocolo_de_ejecucion(self.conn, ejecucion_id)
        guardado = None if previo is None else previo[1]
        if guardado is not None:
            contenido = guardado["manifiesto"]
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
        identidad_esperada = identidad
        if previo is None:
            guardado = protocolo_guardado(self.conn, identidad.run_id)
            if guardado is not None:
                contenido = guardado["manifiesto"]
        else:
            identidad = previo[0]
        if guardado is None and not fitting.m3_skus:
            codigos = sorted({e.code for e in fitting.exclusions if e.code})
            raise ValueError(
                f"ningun SKU llega a M3: {len(fitting.exclusions)} SKU excluidos "
                f"antes de M3 ({', '.join(codigos)})"
            )
        try:
            if guardado is not None and (
                identidad != identidad_esperada
                or guardado["protocolo"] != json.loads(json.dumps(protocolo))
            ):
                raise LoteRechazadoError(
                    "contexto_incompatible", "protocolo durable incompatible"
                )
            if faltaba_manifiesto or (
                guardado is not None
                and ruta_manifiesto.read_text().strip() != contenido.strip()
            ):
                escribir_atomico(ruta_manifiesto, contenido + "\n")
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
                    exclusiones_m2=guardado["procedencia"]["exclusiones_m2"],
                    exclusiones=guardado["procedencia"]["exclusiones_m3"],
                    exclusiones_datos=guardado["procedencia"]["exclusiones_datos"],
                    publicada_previamente=True,
                    datos_sinteticos=datos_sinteticos,
                )
            existente = json.loads(contenido)
            esperados = existente.get("candidatos")
            reconstruidos = json.loads(nuevo)["candidatos"]
            if guardado is not None:
                # Un fallo de ajuste, del intento original o del reintento, no
                # cambia la receta ya aceptada, ni la exclusion al cerrar L2 que
                # ese fallo provoca (ADR-023). Los demas ajustes deben coincidir.
                fallidos = {(u.sku_id, u.family) for u in fitting.units if u.failed}
                fallidos |= {
                    (f["sku"], f["familia"])
                    for f in guardado["procedencia"]["fallos_m2"]
                }
                excluidos = {e.sku_id for e in fitting.exclusions} | {
                    e["sku_id"] for e in guardado["procedencia"]["exclusiones_m2"]
                }
                esperados, reconstruidos = (
                    [
                        c
                        for c in lista
                        if (c["sku"], c["familia"]) not in fallidos
                        and c["sku"] not in excluidos
                    ]
                    for lista in (esperados, reconstruidos)
                )
            if esperados != reconstruidos:
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
        # La procedencia durable manda sobre los fallos y exclusiones del reintento.
        fallos_l2 = {}
        if guardado is not None:
            for fallo in guardado["procedencia"]["fallos_m2"]:
                fallos_l2.setdefault(fallo["sku"], []).append(fallo["causa"])
            exclusiones_m2 = guardado["procedencia"]["exclusiones_m2"]
        else:
            for unidad in fitting.units:
                if unidad.failed:
                    fallos_l2.setdefault(unidad.sku_id, []).append(unidad.record.error)
            exclusiones_m2 = [asdict(e) for e in fitting.exclusions]
        # Un SKU que salio al cerrar L2 conserva su unidad fallida en la procedencia,
        # pero no recibe veredicto: no es un FALLO_TECNICO (ADR-023).
        for exclusion in exclusiones_m2:
            fallos_l2.pop(exclusion["sku_id"], None)
        run_id = registrar_corrida(
            self.conn,
            identidad,
            handoff,
            skus_fallidos=tuple(fallos_l2),
            ingesta_sha256=ingesta_sha,
            datos_sinteticos=datos_sinteticos,
        )
        plan = ventanas_reserva(corte, config.horizonte)
        skus = sorted(
            {c.sku for c in handoff.candidatos}
            | {f.sku for f in handoff.fallos}
            | set(fallos_l2)
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
        # La seleccion de robustez solo lee historia hasta t*: queda fija con el
        # protocolo y una reanudacion la lee de la procedencia durable, aunque
        # el reintento haya perdido ajustes L2 (ADR-023).
        seleccion_previa = (
            guardado["procedencia"]["seleccion_previa"]
            if guardado is not None
            else [
                asdict(s)
                for s in seleccionar_previa(
                    _previas(evaluation, handoff, clases, corte),
                    {clases[sku] for sku in skus},
                    self.politica,
                )
            ]
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
                "seleccion_previa": seleccion_previa,
                "fallos_m2": [
                    {"sku": u.sku_id, "familia": u.family, "causa": u.record.error}
                    for u in fitting.units
                    if u.failed
                ],
                "discrepancia_metrica_hpo": evaluation.settings.metric != "rmse",
                "limite": (
                    "seleccion y puntaje en bloques disjuntos de la misma reserva; "
                    "evidencia retrospectiva"
                ),
            },
            reserva_abierta_en=apertura,
            n_unidades_previstas=n_previstas,
            t_star=corte.t_star,
            lanzamientos_tardios=fitting.late_launches,
            # Los lanzamientos tardios de M1 ya tienen su tabla.
            exclusiones=[
                SkuExclusion(**e)
                for e in exclusiones_m2
                if e["sku_id"] not in {t.sku_id for t in fitting.late_launches}
            ],
            clases=clases,
            ejecucion_id=ejecucion_id,
        )
        # La reserva solo se lee despues del handoff validado y protocolo durable.
        if estado_corrida(self.conn, run_id) in ("EN_CURSO", "EVALUADA"):
            self._pronosticar(
                run_id, handoff, skus, clases, series_sku, planes, corte, config
            )
            if estado_corrida(self.conn, run_id) == "EN_CURSO":
                marcar_evaluada(self.conn, run_id)
            # Cada SKU se lee de SQLite al evaluarlo: en memoria quedan sus
            # metricas, no sus pronosticos.
            fallos_sku = Counter(f.sku for f in handoff.fallos)
            # Solo los SKU sin ningun ajuste L2 exitoso; un candidato que 3.2
            # rechaza ya cuenta como fallo del handoff.
            sin_candidatos = set(fallos_l2) - {
                c["sku"] for c in existente["candidatos"]
            }
            for sku in sin_candidatos:
                fallos_sku[sku] += len(fallos_l2[sku])
            evaluaciones = []
            observaciones = {}
            for sku in skus:
                historia, reserva = separar_reserva(series_sku[sku], corte)
                # El calendario completo de la reserva: un dia sin dato es NULL.
                observaciones[sku] = reserva.reindex(corte.reserved_dates)
                entrada = EntradaSku(
                    sku,
                    clases[sku],
                    historia,
                    reserva,
                    tuple(
                        cargar_serie(
                            self.conn,
                            run_id,
                            sku,
                            candidato.candidato_id,
                            candidato.familia,
                            candidato.modelo,
                        )
                        for candidato in handoff.candidatos
                        if candidato.sku == sku
                    ),
                    fallos_sku[sku]
                    + len(candidatos_con_fallos(self.conn, run_id, sku)),
                )
                base = cargar_serie(
                    self.conn,
                    run_id,
                    sku,
                    _id_linea_base(sku),
                    FAMILIA_LINEA_BASE,
                    MODELO_LINEA_BASE,
                )
                evaluaciones.append(
                    evaluar_bloques(entrada, corte, bloques, linea_base=base)
                )
            resultado = emitir_veredictos(
                evaluaciones,
                bloques,
                datos_sinteticos=datos_sinteticos,
                politica=self.politica,
                seleccion_previa=tuple(
                    SeleccionCategoria(
                        **{**s, "skus_comparables": tuple(s["skus_comparables"])}
                    )
                    for s in seleccion_previa
                ),
            )
            resultado = replace(
                resultado,
                skus=tuple(
                    replace(
                        veredicto,
                        justificacion=veredicto.justificacion
                        + ": "
                        + "; ".join(str(causa) for causa in fallos_l2[veredicto.sku]),
                    )
                    if veredicto.sku in sin_candidatos
                    and veredicto.veredicto == "FALLO_TECNICO"
                    else veredicto
                    for veredicto in resultado.skus
                ),
            )
            guardar_evaluacion(
                self.conn,
                run_id,
                evaluaciones,
                resultado,
                observaciones=observaciones,
            )
        return self._artefacto(
            run_id,
            config,
            n_previstas=n_previstas,
            exclusiones_m2=exclusiones_m2,
            exclusiones=exclusiones,
            exclusiones_datos=exclusiones_datos,
            publicada_previamente=False,
            datos_sinteticos=datos_sinteticos,
        )

    def _pronosticar(
        self,
        run_id: str,
        handoff: HandoffValidado,
        skus: Sequence[str],
        clases: Mapping[str, SkuClass],
        series: Mapping[str, pd.Series],
        planes: Mapping[str, tuple[VentanaReserva, ...]],
        corte: ReserveCut,
        config: ConfiguracionPronosticos,
    ) -> None:
        """Pronosticos de cada SKU en el pool; este proceso es el unico escritor.

        Las unidades nuevas de los candidatos de un SKU se confirman con sus
        marcadores en una sola transaccion, y su linea base en otra: un commit
        por ventana haria de esta escritura el cuello de botella. Los SKU se
        confirman en orden: tras una caida quedan durables los SKU previos y la
        reanudacion parte de sus marcadores.
        """
        en_curso = estado_corrida(self.conn, run_id) == "EN_CURSO"
        fabricas = dict(self.fabricas)

        def tareas() -> Iterator[_TareaSku]:
            for sku in skus:
                plan = planes[sku]
                candidatos = []
                for candidato in (c for c in handoff.candidatos if c.sku == sku):
                    confirmadas: tuple[UnidadPronostico, ...] = ()
                    if plan:
                        confirmadas = cargar_unidades(self.conn, run_id, candidato)
                        if not en_curso and len(confirmadas) != len(plan):
                            raise EvidenciaInconsistenteError(
                                "faltan unidades en una corrida EVALUADA"
                            )
                    candidatos.append((candidato, confirmadas))
                base = None
                if en_curso:
                    base = cargar_serie(
                        self.conn,
                        run_id,
                        sku,
                        _id_linea_base(sku),
                        FAMILIA_LINEA_BASE,
                        MODELO_LINEA_BASE,
                    ).pronosticos
                yield _TareaSku(
                    sku,
                    clases[sku],
                    series[sku],
                    corte,
                    config,
                    fabricas,
                    tuple(candidatos),
                    plan,
                    base,
                )

        def etiqueta(candidato: Candidato) -> EtiquetaUnidad:
            return EtiquetaUnidad(
                "L4", candidato.sku, candidato.familia, candidato.modelo
            )

        # Solo este proceso escribe el progreso de L4: no hay escrituras concurrentes.
        if self.progreso is not None:
            emitir_evento(
                self.progreso,
                "planned",
                stage="L4",
                units=[
                    {k: v for k, v in etiqueta(c).como_dict().items() if k != "stage"}
                    for c in handoff.candidatos
                ],
            )
        with closing(
            iterar_unidades(_pronosticar_sku, tareas(), procesos=self.procesos)
        ) as resultados:
            for sku, resultado in zip(skus, resultados, strict=True):
                generado = resultado.valor
                if generado is None:
                    raise RuntimeError(
                        f"L4 sin pronosticos sku={sku}: {resultado.registro.error}"
                    )
                guardar_unidades_generadas(self.conn, run_id, generado.nuevas)
                if generado.linea_base is not None:
                    id_base = _id_linea_base(sku)
                    guardar_unidades(
                        self.conn,
                        run_id,
                        sku,
                        generado.linea_base,
                        codigo_version=self.codigo_version,
                        semilla=0,
                        emitido_en=datetime.now(UTC),
                        fallos={
                            p.origen: FalloPronostico(
                                sku,
                                id_base,
                                MODELO_LINEA_BASE,
                                p.origen,
                                "valores",
                                "pronostico_invalido",
                                f"sku={sku} linea base sin pronostico valido",
                            )
                            for p in generado.linea_base.pronosticos
                            if not np.isfinite(p.valores).all()
                        },
                    )
                if self.progreso is not None:
                    # El SKU ya es durable: se anuncia el fin de cada candidato suyo.
                    for candidato in (c for c in handoff.candidatos if c.sku == sku):
                        emitir_evento(
                            self.progreso,
                            "unit_finished",
                            **etiqueta(candidato).como_dict(),
                            state="completada",
                        )

    def _artefacto(
        self,
        run_id: str,
        config: ConfiguracionPronosticos,
        *,
        n_previstas: int,
        exclusiones_m2: Sequence[Mapping[str, Any]],
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
                # SKU fuera de M3 segun la procedencia de la publicacion: en una
                # reanudacion manda sobre las exclusiones del reintento.
                "exclusiones_m2": list(exclusiones_m2),
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


def _id_linea_base(sku: str) -> str:
    return f"{sku}/{FAMILIA_LINEA_BASE}/{MODELO_LINEA_BASE}"


def _previas(
    evaluation: EvaluationArtifact,
    handoff: HandoffValidado,
    clases: Mapping[str, SkuClass],
    corte: ReserveCut,
) -> tuple[EvaluacionSku, ...]:
    """Pliegues L3 de cada candidato del handoff, para la seleccion de robustez."""
    ids = {(c.sku, c.familia): c for c in handoff.candidatos}
    por_sku: dict[str, tuple[np.ndarray, list[CandidatoPrevio]]] = {}
    for evaluado in evaluation.candidates:
        seleccion = evaluado.fitted.selection
        candidato = ids.get((seleccion.sku_id, seleccion.family))
        if candidato is None:
            continue
        _, propios = por_sku.setdefault(candidato.sku, (evaluado.fitted.series, []))
        propios.append(
            CandidatoPrevio(
                candidato.candidato_id,
                candidato.familia,
                candidato.modelo,
                evaluado.walk_forward,
            )
        )
    return tuple(
        evaluacion_previa(sku, clases[sku], corte.t_star, historia, candidatos)
        for sku, (historia, candidatos) in sorted(por_sku.items())
    )


@dataclass(frozen=True, slots=True)
class _TareaSku:
    """Lo que un proceso del pool necesita para pronosticar un SKU sin SQLite.

    ``base_confirmada`` es ``None`` cuando la linea base ya no se genera
    (corrida EVALUADA).
    """

    sku: str
    sku_class: SkuClass
    serie: pd.Series
    corte: ReserveCut
    configuracion: ConfiguracionPronosticos
    fabricas: Mapping[str, FabricaAdaptador]
    candidatos: tuple[tuple[Candidato, tuple[UnidadPronostico, ...]], ...]
    plan: tuple[VentanaReserva, ...]
    base_confirmada: tuple[PronosticoFechado, ...] | None


@dataclass(frozen=True, slots=True)
class _PronosticosSku:
    nuevas: tuple[UnidadPronostico, ...]
    linea_base: SerieCandidato | None


def _pronosticar_sku(tarea: _TareaSku) -> _PronosticosSku:
    """Unidad del pool: pronosticos nuevos de un SKU, sin escribir en SQLite."""
    nuevas: list[UnidadPronostico] = []
    series = []
    for candidato, confirmadas in tarea.candidatos:
        pronosticos: tuple[PronosticoFechado, ...] = ()
        if tarea.plan:
            generado = generar_pronosticos(
                candidato,
                tarea.serie,
                tarea.corte,
                tarea.configuracion,
                fabricas=tarea.fabricas,
                confirmadas=confirmadas,
                confirmar=nuevas.append,
            )
            pronosticos = tuple(u.pronostico for u in generado.unidades)
        series.append(
            SerieCandidato(
                candidato.candidato_id, candidato.familia, candidato.modelo, pronosticos
            )
        )
    if tarea.base_confirmada is None:
        return _PronosticosSku(tuple(nuevas), None)
    historia, reserva = separar_reserva(tarea.serie, tarea.corte)
    base = pronosticar_linea_base(
        EntradaSku(tarea.sku, tarea.sku_class, historia, reserva, tuple(series)),
        tarea.corte,
        ventanas_planeadas={v.origen: v.fechas for v in tarea.plan},
        confirmados=tarea.base_confirmada,
    )
    return _PronosticosSku(tuple(nuevas), base)
