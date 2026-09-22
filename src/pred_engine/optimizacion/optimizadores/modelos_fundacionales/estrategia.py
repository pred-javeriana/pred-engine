"""`FoundationSelectionStrategy`: la familia fundacional como estrategia del router.

A diferencia de `MLSelectionStrategy`, aqui no hay nada que seleccionar: el
modelo se usa zero-shot con su configuracion congelada. `select()`
no carga el modelo, no mira la serie, no corre HPO ni Walk-Forward; solo
entrega la configuracion base. La evaluacion (errores, Diebold-Mariano) es del
Modulo 3. El router no importa este modulo: la estrategia se inyecta al
registrarla.
"""

from __future__ import annotations

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos.modelos_fundacionales.configuracion import (
    CHRONOS2_ZERO_SHOT,
)
from pred_engine.optimizacion.router.contratos import (
    PredictorFamily,
    SelectionRequest,
    SelectionResult,
    TopologicalProfile,
)

_logger = get_logger(__name__)

FAMILIA_FUNDACIONAL: PredictorFamily = "foundation"


class FoundationSelectionStrategy:
    family: PredictorFamily = FAMILIA_FUNDACIONAL

    def select(
        self, request: SelectionRequest, profile: TopologicalProfile
    ) -> SelectionResult:
        # El perfil solo anota el resultado: la configuracion es la misma para
        # las cuatro clases de SKU.
        _logger.info(
            "Configuracion fundacional sku_id=%s perfil=%s modelo=%s@%s",
            request.sku_id,
            profile,
            CHRONOS2_ZERO_SHOT.model_id,
            CHRONOS2_ZERO_SHOT.revision,
        )
        return SelectionResult(
            sku_id=request.sku_id,
            sku_class=request.sku_class,
            family=FAMILIA_FUNDACIONAL,
            profile=profile,
            produced_by=type(self).__name__,
            payload={
                "configuracion": CHRONOS2_ZERO_SHOT.descripcion_canonica(),
                "optimizado": False,
            },
        )
