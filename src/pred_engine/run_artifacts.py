"""Persistencia de una corrida L1-L3 en su propio directorio.

El directorio de la corrida reune lo que M3 y la tesis consumen sin volver a
ejecutar nada: el manifiesto de candidatos que valida el adaptador de M3
(ADR-03-004), los pronosticos desde t*, la evidencia walk-forward, la traza de
cada unidad (proceso, inicio y fin) y un resumen ``corrida.json`` con la
procedencia de cada candidato (estudio HPO y versiones). Los estudios de HPO
persisten aparte, bajo ``hpo/``, con su propio control de reanudacion (2.9).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from pred_engine.comun.ejecucion_paralela import resumen_paralelismo
from pred_engine.comun.modelos.manifiesto_candidatos import (
    MODELO_POR_FAMILIA,
    ContextoParticion,
)
from pred_engine.optimizacion.control_reanudacion.almacenamiento import (
    escribir_atomico,
)
from pred_engine.optimizacion.router import SelectionResult, construir_manifiesto
from pred_engine.pipeline import FittingArtifact, PipelineResult, summarize_run

RUN_SCHEMA = "pred-engine.corrida/1"
_PAQUETES = (
    "pred-engine",
    "numpy",
    "pandas",
    "statsmodels",
    "lightgbm",
    "optuna",
    "pydantic",
    "chronos-forecasting",
    "torch",
)
_EVIDENCIA_HPO = (
    "metrica_objetivo",
    "valor",
    "n_ventanas",
    "n_trials",
    "n_completados",
    "n_podados",
    "n_fallidos",
)


def library_versions() -> dict[str, str]:
    """Versiones que fijan la reproducibilidad de los candidatos."""
    versiones = {"python": platform.python_version()}
    for paquete in _PAQUETES:
        try:
            versiones[paquete] = version(paquete)
        except PackageNotFoundError:
            continue
    return versiones


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def code_fingerprint() -> str:
    """Huella del codigo fuente de pred_engine instalado.

    Forma parte de la identidad de la corrida: con otro codigo, los estudios
    HPO persistidos no se reutilizan aunque la entrada y las opciones coincidan.
    """
    raiz = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for ruta in sorted(raiz.rglob("*.py")):
        digest.update(ruta.relative_to(raiz).as_posix().encode("utf-8"))
        digest.update(ruta.read_bytes())
    return digest.hexdigest()[:16]


def run_identifier(input_sha256: str, settings: Mapping[str, Any]) -> str:
    """Identidad estable: misma entrada y configuracion, misma corrida."""
    huella = json.dumps(
        {"input": input_sha256, "settings": settings}, sort_keys=True, default=str
    )
    return "r-" + hashlib.sha256(huella.encode("utf-8")).hexdigest()[:16]


def _contexto(fitting: FittingArtifact) -> ContextoParticion:
    """Particion de 3.1 con la que M2 optimizo; M3 la exige identica."""
    return ContextoParticion(
        ingesta_ref_m1=sha256_file(fitting.ingestion.parquet_path),
        t_corte_reserva=fitting.reserve.t_star.date(),
        fraccion_reserva=fitting.reserve.fraction,
    )


def _procedencia(candidato_id: str, seleccion: SelectionResult) -> dict[str, Any]:
    """Lo que el manifiesto no lleva: politica, perfil y evidencia del HPO."""
    payload = seleccion.payload
    return {
        "candidato_id": candidato_id,
        "sku_id": seleccion.sku_id,
        "familia": seleccion.family,
        "perfil": seleccion.profile,
        "politica": seleccion.policy_version,
        "estudio_hpo": payload.get("estudio_hpo"),
        "evidencia_hpo": {c: payload[c] for c in _EVIDENCIA_HPO if c in payload},
    }


def _instante(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).isoformat()


def _escribir_parquet(frame: pd.DataFrame, ruta: Path) -> None:
    temporal = ruta.with_name(ruta.name + ".tmp")
    frame.to_parquet(temporal, index=False)
    os.replace(temporal, ruta)


def _escribir_json(datos: Any, ruta: Path) -> None:
    escribir_atomico(ruta, json.dumps(datos, ensure_ascii=False, indent=2) + "\n")


def write_run(
    result: PipelineResult,
    directory: str | Path,
    *,
    run_id: str,
    metadata: Mapping[str, Any],
) -> dict[str, Path]:
    """Escribe los artefactos disponibles de ``result`` y devuelve sus rutas.

    Una corrida que fallo en L2 deja solo ``corrida.json`` y ``unidades.jsonl``;
    repetirla con el mismo ``run_id`` reescribe estos archivos de forma atomica.
    """
    destino = Path(directory)
    destino.mkdir(parents=True, exist_ok=True)
    archivos: dict[str, Path] = {}

    unidades = destino / "unidades.jsonl"
    lineas = [
        json.dumps(
            {
                "stage": unit.stage,
                "sku_id": unit.sku_id,
                "family": unit.family,
                "state": unit.record.estado,
                "pid": unit.record.pid,
                "started_at": _instante(unit.record.inicio),
                "finished_at": _instante(unit.record.fin),
                "seconds": round(unit.record.duracion_s, 3),
                "error": unit.record.error,
            },
            ensure_ascii=False,
        )
        for unit in result.units
    ]
    escribir_atomico(unidades, "".join(f"{linea}\n" for linea in lineas))
    archivos["unidades"] = unidades

    fitting, evaluation = result.fitting, result.evaluation
    procedencia: list[dict[str, Any]] = []
    if fitting is not None:
        selecciones = [c.selection for c in fitting.candidates]
        manifiesto = construir_manifiesto(
            selecciones,
            run_id_m2=run_id,
            contexto=_contexto(fitting),
            emitido_en=datetime.now(UTC),
        )
        archivos["candidatos"] = destino / "candidatos.json"
        escribir_atomico(archivos["candidatos"], manifiesto + "\n")
        procedencia = [
            _procedencia(crudo["candidato_id"], seleccion)
            for crudo, seleccion in zip(
                json.loads(manifiesto)["candidatos"], selecciones, strict=True
            )
        ]
        fechas = fitting.reserve.reserved_dates
        archivos["pronosticos"] = destino / "pronosticos.parquet"
        _escribir_parquet(
            pd.DataFrame(
                [
                    {
                        "sku_id": c.selection.sku_id,
                        "sku_class": c.selection.sku_class,
                        "familia": c.selection.family,
                        "modelo": MODELO_POR_FAMILIA[c.selection.family],
                        "timestamp": fecha,
                        "pronostico": float(valor),
                    }
                    for c in fitting.candidates
                    for fecha, valor in zip(fechas, c.forecast, strict=True)
                ]
            ),
            archivos["pronosticos"],
        )

    if fitting is not None and evaluation is not None:
        resumen, ventanas = [], []
        t_star = fitting.reserve.t_star
        for evaluado in evaluation.candidates:
            seleccion = evaluado.fitted.selection
            evidencia = evaluado.walk_forward
            inicio = t_star - pd.Timedelta(days=len(evaluado.fitted.series) - 1)
            metricas = pd.DataFrame([dict(v.metricas) for v in evidencia.ventanas])
            resumen.append(
                {
                    "sku_id": seleccion.sku_id,
                    "sku_class": seleccion.sku_class,
                    "familia": seleccion.family,
                    "modelo": MODELO_POR_FAMILIA[seleccion.family],
                    "metrica": evidencia.metrica_objetivo,
                    "valor_agregado": float(evidencia.valor_agregado),
                    "n_ventanas": evidencia.n_ventanas_evaluadas,
                    **{f"{k}_media": float(v) for k, v in metricas.mean().items()},
                }
            )
            for resultado in evidencia.ventanas:
                ventana = resultado.ventana
                for paso, (real, pronostico) in enumerate(
                    zip(resultado.y_real, resultado.y_pred, strict=True)
                ):
                    ventanas.append(
                        {
                            "sku_id": seleccion.sku_id,
                            "familia": seleccion.family,
                            "ventana": ventana.indice,
                            "timestamp": inicio
                            + pd.Timedelta(days=ventana.inicio_val + paso),
                            "real": float(real),
                            "pronostico": float(pronostico),
                        }
                    )
        archivos["evaluacion"] = destino / "evaluacion.parquet"
        _escribir_parquet(pd.DataFrame(resumen), archivos["evaluacion"])
        archivos["walk_forward"] = destino / "walk_forward.parquet"
        _escribir_parquet(pd.DataFrame(ventanas), archivos["walk_forward"])

    corrida = destino / "corrida.json"
    archivos["corrida"] = corrida
    _escribir_json(
        {
            "schema": RUN_SCHEMA,
            "run_id": run_id,
            "written_at": datetime.now(UTC).isoformat(),
            **metadata,
            "summary": summarize_run(result),
            "versions": library_versions(),
            "candidates": procedencia,
            "parallelism": {
                "L2": resumen_paralelismo(
                    [u.record for u in (fitting.units if fitting else ())]
                ),
                "L3": resumen_paralelismo(
                    [u.record for u in (evaluation.units if evaluation else ())]
                ),
            },
            "files": {nombre: ruta.name for nombre, ruta in sorted(archivos.items())},
        },
        corrida,
    )
    return archivos
