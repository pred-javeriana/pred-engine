"""Enrutador categorico de la matriz Syntetos-Boylan."""

from __future__ import annotations

import math

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos import ADI_THRESHOLD, CV2_THRESHOLD, SkuClass
from pred_engine.ingesta.categorizacion.errores import TopologyRoutingError

_logger = get_logger(__name__)


def route_syntetos_boylan(adi: float, cv2: float) -> SkuClass:
    """Clasifica un SKU. Empates en el umbral caen en Intermittent/Erratic/Lumpy."""
    if not math.isfinite(adi) or not math.isfinite(cv2):
        raise TopologyRoutingError("ADI y CV2 deben ser finitos")
    if adi <= 0.0 or cv2 < 0.0:
        raise TopologyRoutingError("ADI debe ser > 0 y CV2 debe ser >= 0")

    adi_alto = adi >= ADI_THRESHOLD
    cv2_alto = cv2 >= CV2_THRESHOLD

    if not adi_alto and not cv2_alto:
        clase: SkuClass = "smooth"
    elif adi_alto and not cv2_alto:
        clase = "intermittent"
    elif not adi_alto and cv2_alto:
        clase = "erratic"
    else:
        clase = "lumpy"

    _logger.info(
        "Enrutado Syntetos-Boylan adi=%.6f cv2=%.6f clase=%s",
        adi,
        cv2,
        clase,
    )
    return clase
