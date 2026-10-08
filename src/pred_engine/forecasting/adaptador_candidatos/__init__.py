"""Adaptador del conjunto de candidatos de M2 (3.2): valida e instancia.

No ajusta ni pronostica (3.3), no evalua (3.4) y no persiste (3.5): entrega
el ganador del HPO de cada familia (uno por familia y SKU) que valido, y los
fallos. La eleccion entre familias es de M3.
"""

from pred_engine.forecasting.adaptador_candidatos.contratos import (
    AdaptadorCandidato,
    CodigoFallo,
    FabricaAdaptador,
    FalloCandidato,
    HandoffValidado,
    MotivoRechazo,
)
from pred_engine.forecasting.adaptador_candidatos.errores import (
    AdaptadorCandidatosError,
    LoteRechazadoError,
    ModeloNoRegistradoError,
)
from pred_engine.forecasting.adaptador_candidatos.registro import (
    FABRICAS,
    PERIODO_LINEA_BASE,
    instanciar,
    instanciar_linea_base,
)
from pred_engine.forecasting.adaptador_candidatos.validacion import (
    validar_manifiesto,
)

__all__ = [
    "AdaptadorCandidato",
    "AdaptadorCandidatosError",
    "CodigoFallo",
    "FABRICAS",
    "FabricaAdaptador",
    "FalloCandidato",
    "HandoffValidado",
    "LoteRechazadoError",
    "ModeloNoRegistradoError",
    "MotivoRechazo",
    "PERIODO_LINEA_BASE",
    "instanciar",
    "instanciar_linea_base",
    "validar_manifiesto",
]
