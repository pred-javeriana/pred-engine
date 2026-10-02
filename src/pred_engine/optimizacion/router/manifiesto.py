"""Productor M2 del manifiesto de candidatos (ADR-03-004).

A M3 pasa solo la configuracion GANADORA del HPO de cada familia (una por
familia y SKU), nunca los demas trials. M2 no elige entre familias: eso lo
hace M3. Usa `forecast_config` (el mejor trial, en el formato de la fabrica),
no `payload` (evidencia de la busqueda).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos.manifiesto_candidatos import (
    MODELO_POR_FAMILIA,
    VERSION_ESQUEMA_MANIFIESTO,
    ContextoParticion,
    ManifiestoCandidatos,
    configuracion_declarada,
)
from pred_engine.optimizacion.router.contratos import SelectionResult
from pred_engine.optimizacion.router.errores import SelectionContractError

_logger = get_logger(__name__)


def construir_manifiesto(
    resultados: Sequence[SelectionResult],
    *,
    run_id_m2: str,
    contexto: ContextoParticion,
    emitido_en: datetime,
) -> str:
    """JSON del manifiesto: un candidato por `SelectionResult`, es decir, el
    ganador del HPO de cada familia enrutada para cada SKU."""
    manifiesto = ManifiestoCandidatos(
        schema_version=VERSION_ESQUEMA_MANIFIESTO,
        run_id_m2=run_id_m2,
        emitido_en=emitido_en,
        contexto=contexto,
        candidatos=tuple(_candidato(resultado) for resultado in resultados),
    )
    _logger.info(
        "Manifiesto M2 emitido run_id_m2=%s candidatos=%d",
        run_id_m2,
        len(manifiesto.candidatos),
    )
    return manifiesto.model_dump_json()


def _candidato(resultado: SelectionResult) -> dict[str, Any]:
    if resultado.forecast_config is None:
        _logger.error(
            "Resultado sin forecast_config sku_id=%s familia=%s",
            resultado.sku_id,
            resultado.family,
        )
        raise SelectionContractError(
            f"la estrategia {resultado.produced_by} no entrego forecast_config "
            f"para sku_id={resultado.sku_id!r}"
        )
    modelo = MODELO_POR_FAMILIA[resultado.family]
    return {
        "candidato_id": f"{resultado.sku_id}/{resultado.family}/{modelo}",
        "sku": resultado.sku_id,
        "sku_class": resultado.sku_class,
        "familia": resultado.family,
        "modelo": modelo,
        "semilla": resultado.forecast_seed,
        "configuracion": configuracion_declarada(
            resultado.family, resultado.forecast_config
        ),
    }
