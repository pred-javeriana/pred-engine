"""Seleccion por SKU de una corrida: ``seleccion.jsonl`` (``pred-engine.seleccion/1``).

Una linea por SKU con lo que la plataforma necesita para mostrar la seleccion
sin volver a calcular nada: por cada una de las cuatro familias, su estado, la
evidencia de la configuracion elegida, los ensayos del HPO (completados, podados
y fallidos, con su motivo) y el Walk-Forward por ventana. El contenido sigue la
forma de ``FamilyEntry`` del contrato ``sku_selection``; lo que el contrato
normaliza con vocabulario de la plataforma (el codigo de un error) se entrega
crudo -- clase y mensaje de la excepcion, causa de la exclusion -- para que no
se defina en dos lugares.

Solo se escribe lo que la corrida produjo: sin L2 no hay seleccion. Los ensayos
se leen de ``hpo/`` sin Optuna (``comun.ensayos_hpo``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from pred_engine.comun.ensayos_hpo import (
    buscar_estudio,
    ensayo_a_dict,
    leer_ensayos,
)
from pred_engine.comun.modelos.manifiesto_candidatos import MODELO_POR_FAMILIA
from pred_engine.optimizacion.control_reanudacion.almacenamiento import (
    escribir_atomico,
)
from pred_engine.optimizacion.router.contratos import (
    PREDICTOR_FAMILIES,
    PredictorFamily,
    RoutingPolicy,
)
from pred_engine.optimizacion.router.politica import (
    FAMILIES_BY_SKU_CLASS,
    INITIAL_POLICY_VERSION,
    PROFILE_BY_SKU_CLASS,
    InitialTopologyPolicy,
)
from pred_engine.pipeline import (
    EvaluatedCandidate,
    FittedCandidate,
    PipelineResult,
    UnitExecution,
)

SELECCION_SCHEMA = "pred-engine.seleccion/1"
ARCHIVO_SELECCION = "seleccion.jsonl"


def _limpio(valor: Any) -> Any:
    """JSON estricto: un numero no finito se vuelve nulo en vez de ``NaN``."""
    if isinstance(valor, float):
        return valor if math.isfinite(valor) else None
    if isinstance(valor, Mapping):
        return {str(k): _limpio(v) for k, v in valor.items()}
    if isinstance(valor, list | tuple):
        return [_limpio(v) for v in valor]
    return valor


def _modelo(familia: str) -> str:
    return f"{familia}:{MODELO_POR_FAMILIA[familia]}"


def _fallo(
    unidad: UnitExecution | None, etapa: str, texto: str | None
) -> dict[str, str]:
    """``texto`` es lo que registra la unidad: ``NombreDeClase: mensaje``."""
    if unidad is not None and unidad.record.error:
        texto = unidad.record.error
        etapa = unidad.stage
    return {"stage": etapa, "text": texto or "La unidad fallo sin detalle."}


def _resumen_walk_forward(
    evaluado: EvaluatedCandidate, inicio: pd.Timestamp, *, scope: str, comparable: bool
) -> dict[str, Any]:
    evidencia = evaluado.walk_forward
    return {
        "metrica_objetivo": evidencia.metrica_objetivo,
        "valor_agregado": float(evidencia.valor_agregado),
        "completo": evidencia.completo,
        "n_ventanas_evaluadas": evidencia.n_ventanas_evaluadas,
        "n_ventanas_totales": evidencia.n_ventanas_totales,
        "selection_scope": scope,
        "statistical_comparison_available": comparable,
        "windows": [
            {
                "indice": r.ventana.indice,
                # Primer dia de validacion de la ventana.
                "corte": (inicio + pd.Timedelta(days=r.ventana.inicio_val))
                .date()
                .isoformat(),
                "inicio_train": r.ventana.inicio_train,
                "fin_train": r.ventana.fin_train,
                "inicio_val": r.ventana.inicio_val,
                "fin_val": r.ventana.fin_val,
                "metricas": {k: float(v) for k, v in r.metricas.items()},
                "duracion_s": round(r.duracion_s, 3),
                "fallo": r.fallo,
            }
            for r in evidencia.ventanas
        ],
    }


class _Contexto:
    """Lo de la corrida que comparten todas las familias de todos los SKU."""

    def __init__(self, result: PipelineResult, hpo: Path) -> None:
        fitting = result.fitting
        assert fitting is not None
        self.hpo = hpo
        self.fitting = fitting
        self.evaluation = result.evaluation
        self.politica: RoutingPolicy = fitting.policy or InitialTopologyPolicy()
        self.t_star = fitting.reserve.t_star
        self.excluidos = {e.sku_id: e.cause for e in fitting.exclusions}
        self.candidatos: dict[tuple[str, str], FittedCandidate] = {
            (c.selection.sku_id, c.selection.family): c for c in fitting.candidates
        }
        self.evaluados: dict[tuple[str, str], EvaluatedCandidate] = (
            {
                (c.fitted.selection.sku_id, c.fitted.selection.family): c
                for c in self.evaluation.candidates
            }
            if self.evaluation
            else {}
        )
        self.fallos_l2 = self._fallos(fitting.units)
        self.fallos_l3 = self._fallos(self.evaluation.units if self.evaluation else ())
        etapa_l3 = next((s for s in result.stages if s.definition.id == "L3"), None)
        # Si toda L3 fallo no hay unidades, y la causa esta en el mensaje de la etapa.
        self.mensaje_l3 = etapa_l3.message if etapa_l3 is not None else None

    @staticmethod
    def _fallos(unidades: Iterable[UnitExecution]) -> dict[tuple[str, str], Any]:
        return {(u.sku_id, u.family): u for u in unidades if u.failed}

    def ensayos(
        self, familia: str, sku_id: str, estudio: str | None
    ) -> list[dict[str, Any]] | None:
        """Ensayos del estudio de la unidad; ``None`` si no hay estudio persistido."""
        carpeta = self.hpo / estudio if estudio else None
        if carpeta is None or not carpeta.is_dir():
            carpeta = buscar_estudio(self.hpo, familia, sku_id)
        if carpeta is None:
            return None
        return [ensayo_a_dict(t) for t in leer_ensayos(carpeta)]


def _evidencia(seleccion_familia: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {"family": seleccion_familia, **payload}


def _familia(
    ctx: _Contexto, sku_id: str, sku_class: str, familia: PredictorFamily
) -> dict[str, Any]:
    permitidas = {d.family for d in ctx.politica.decide(sku_class)}  # type: ignore[arg-type]
    if familia not in permitidas:
        if familia not in FAMILIES_BY_SKU_CLASS[sku_class]:  # type: ignore[index]
            razon, detalle = (
                "not_in_policy_matrix",
                f"La politica {INITIAL_POLICY_VERSION} no asigna {familia} "
                f"a la clase {sku_class}.",
            )
        else:
            configuradas = getattr(ctx.politica, "families", None) or tuple(permitidas)
            plural = len(configuradas) != 1
            razon, detalle = (
                "not_configured",
                "La corrida se configuro solo con "
                f"{'las familias' if plural else 'la familia'} "
                f"{', '.join(configuradas)}.",
            )
        return {
            "family": familia,
            "estado": "excluida",
            "exclusion": {"reason_code": razon, "detail": detalle},
        }

    if sku_id in ctx.excluidos:
        return {
            "family": familia,
            "estado": "no_ejecutable",
            "cause": ctx.excluidos[sku_id],
        }

    clave = (sku_id, familia)
    candidato = ctx.candidatos.get(clave)
    if candidato is None:
        unidad = ctx.fallos_l2.get(clave)
        entrada: dict[str, Any] = {
            "family": familia,
            "estado": "fallida",
            "modelo": _modelo(familia),
            "error": _fallo(unidad, "L2", None),
            "trials": ctx.ensayos(familia, sku_id, None),
        }
        return entrada

    seleccion = candidato.selection
    entrada = {
        "family": familia,
        "modelo": _modelo(familia),
        "produced_by": seleccion.produced_by,
        "evidence": _evidencia(familia, seleccion.payload),
        "forecast_config": dict(seleccion.forecast_config or {}),
        "forecast_seed": seleccion.forecast_seed,
        # La familia fundacional no optimiza: sin estudio, sin ensayos.
        "trials": ctx.ensayos(familia, sku_id, seleccion.payload.get("estudio_hpo"))
        if seleccion.payload.get("optimizado") is not False
        else None,
    }
    evaluado = ctx.evaluados.get(clave)
    if evaluado is None:
        entrada["estado"] = "fallida"
        entrada["error"] = _fallo(ctx.fallos_l3.get(clave), "L3", ctx.mensaje_l3)
        return entrada
    inicio = pd.Timestamp(ctx.t_star) - pd.Timedelta(days=len(candidato.series) - 1)
    entrada["estado"] = "exitosa"
    entrada["walk_forward"] = _resumen_walk_forward(
        evaluado,
        inicio,
        scope=ctx.evaluation.selection_scope,  # type: ignore[union-attr]
        comparable=ctx.evaluation.statistical_comparison_available,  # type: ignore[union-attr]
    )
    return entrada


def construir_seleccion(
    result: PipelineResult, *, run_id: str, hpo: str | Path
) -> list[dict[str, Any]]:
    """Un documento por SKU del panel, o nada si la corrida no llego a ajustar (L2)."""
    fitting = result.fitting
    if fitting is None:
        return []
    ctx = _Contexto(result, Path(hpo))
    clases = (
        fitting.ingestion.panel.groupby("sku_id", sort=True)["sku_class"]
        .first()
        .to_dict()
    )
    documentos = []
    for sku_id, sku_class in clases.items():
        propios = [
            c.selection.policy_version
            for (sku, _), c in ctx.candidatos.items()
            if sku == sku_id
        ]
        documentos.append(
            _limpio(
                {
                    "schema": SELECCION_SCHEMA,
                    "run_id": run_id,
                    "sku_id": str(sku_id),
                    "sku_class": str(sku_class),
                    "profile": PROFILE_BY_SKU_CLASS[sku_class],
                    "policy_version": next(iter(propios), None) or ctx.politica.version,
                    "families": [
                        _familia(ctx, str(sku_id), sku_class, familia)
                        for familia in PREDICTOR_FAMILIES
                    ],
                }
            )
        )
    return documentos


def escribir_seleccion(
    result: PipelineResult, destino: str | Path, *, run_id: str, hpo: str | Path
) -> Path | None:
    """Escribe ``seleccion.jsonl`` de forma atomica; ``None`` si no hay que escribir."""
    documentos = construir_seleccion(result, run_id=run_id, hpo=hpo)
    if not documentos:
        return None
    ruta = Path(destino) / ARCHIVO_SELECCION
    escribir_atomico(
        ruta,
        "".join(
            json.dumps(d, ensure_ascii=False, allow_nan=False) + "\n"
            for d in documentos
        ),
    )
    return ruta
