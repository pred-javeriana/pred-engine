"""Registro familia -> fabrica e instanciacion de candidatos validados.

Usa las mismas fabricas que M2 (las de `pipeline_setup.model_factories`), asi
M3 construye exactamente lo que M2 evaluo. Se importan aqui y no desde
`pipeline_setup` para que el pipeline pueda depender de M3 sin ciclos.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos.manifiesto_candidatos import (
    Candidato,
    CandidatoFundacional,
)
from pred_engine.comun.modelos.modelos_deep_learning import fabrica_dl
from pred_engine.comun.modelos.modelos_fundacionales.chronos2 import (
    fabrica_fundacional,
)
from pred_engine.comun.modelos.modelos_fundacionales.pipeline import (
    PipelineFundacional,
)
from pred_engine.comun.modelos.modelos_machine_learning.lgbm import fabrica_ml
from pred_engine.comun.walkforward.protocolos import (
    FabricaPronosticador,
    Pronosticador,
)
from pred_engine.forecasting.adaptador_candidatos.errores import (
    ModeloNoRegistradoError,
)
from pred_engine.forecasting.base import SeasonalNaiveStub
from pred_engine.optimizacion.optimizadores.modelos_clasicos import fabrica_sarima

_logger = get_logger(__name__)

# Mismo periodo que la escala de MASE (ADR-03-006).
PERIODO_LINEA_BASE = 7

FABRICAS: Mapping[str, FabricaPronosticador] = MappingProxyType(
    {
        "classical": fabrica_sarima,
        "ml": fabrica_ml,
        "dl": fabrica_dl,
        "foundation": fabrica_fundacional,
    }
)


def instanciar(
    candidato: Candidato,
    *,
    fabricas: Mapping[str, FabricaPronosticador] = FABRICAS,
    pipeline_fundacional: PipelineFundacional | None = None,
) -> Pronosticador:
    """Pronosticador sin ajustar con la configuracion validada, sin defaults."""
    if candidato.familia not in fabricas:
        _logger.error(
            "Familia sin fabrica candidato_id=%s familia=%s",
            candidato.candidato_id,
            candidato.familia,
        )
        raise ModeloNoRegistradoError(
            f"la familia {candidato.familia!r} no tiene fabrica registrada"
        )
    if isinstance(candidato, CandidatoFundacional):
        # Configuracion congelada: la fabrica rechaza una configuracion no vacia.
        return fabrica_fundacional(
            {}, seed=candidato.semilla, pipeline=pipeline_fundacional
        )
    return fabricas[candidato.familia](
        candidato.configuracion.model_dump(), seed=candidato.semilla
    )


def instanciar_linea_base() -> Pronosticador:
    """Seasonal Naive: recorre las mismas ventanas que los candidatos."""
    return SeasonalNaiveStub(season_length=PERIODO_LINEA_BASE)
