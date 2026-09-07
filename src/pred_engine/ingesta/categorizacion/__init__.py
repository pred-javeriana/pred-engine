"""Motor de topologia Syntetos-Boylan: ADI, CV² y enrutado de sku_class."""

from pred_engine.ingesta.categorizacion.adi import compute_adi
from pred_engine.ingesta.categorizacion.cv2 import compute_cv2
from pred_engine.ingesta.categorizacion.errores import (
    TopologyContractError,
    TopologyMathError,
    TopologyRoutingError,
)

__all__ = [
    "TopologyContractError",
    "TopologyMathError",
    "TopologyRoutingError",
    "compute_adi",
    "compute_cv2",
]
