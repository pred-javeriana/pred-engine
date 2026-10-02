"""Persistencia de una corrida L1-L3 en su propio directorio.

El directorio de la corrida reune lo que M3 y la tesis consumen sin volver a
ejecutar nada: el manifiesto tipado de candidatos (ADR-03-004), los pronosticos
desde t*, la evidencia walk-forward, la traza de cada unidad (proceso, inicio y
fin) y un resumen ``corrida.json``. Los estudios de HPO persisten aparte, bajo
``hpo/``, con su propio control de reanudacion (2.9).
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
from pred_engine.optimizacion.control_reanudacion.almacenamiento import (
    escribir_atomico,
)
from pred_engine.optimizacion.manifiesto_candidatos import (
    ContextoCorte,
    construir_manifiesto,
)
from pred_engine.pipeline import PipelineResult, ReserveCut, summarize_run

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
_MODELO_POR_FAMILIA = {
    "classical": "sarima",
    "ml": "lightgbm",
    "dl": "mlp",
    "foundation": "chronos-2",
}


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


def _contexto(reserve: ReserveCut) -> ContextoCorte:
    return ContextoCorte(
        t_estrella=reserve.t_star.date(),
        primer_dia_reservado=reserve.first_reserved.date(),
        ultimo_dia_observado=reserve.last_observed.date(),
        dias_reservados=reserve.reserved_days,
        fraccion_reservada=reserve.fraction,
    )


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
    if fitting is not None:
        manifiesto = construir_manifiesto(
            [c.selection for c in fitting.candidates],
            run_id=run_id,
            contexto=_contexto(fitting.reserve),
            versiones=library_versions(),
        )
        archivos["candidatos"] = destino / "candidatos.json"
        escribir_atomico(
            archivos["candidatos"], manifiesto.model_dump_json(indent=2) + "\n"
        )
        fechas = fitting.reserve.reserved_dates
        archivos["pronosticos"] = destino / "pronosticos.parquet"
        _escribir_parquet(
            pd.DataFrame(
                [
                    {
                        "sku_id": c.selection.sku_id,
                        "sku_class": c.selection.sku_class,
                        "familia": c.selection.family,
                        "modelo": _MODELO_POR_FAMILIA[c.selection.family],
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
                    "modelo": _MODELO_POR_FAMILIA[seleccion.family],
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
