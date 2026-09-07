"""Motor de topologia Syntetos-Boylan: ADI, CV² y enrutado de sku_class."""

from pred_engine.ingesta.categorizacion.adi import compute_adi
from pred_engine.ingesta.categorizacion.cv2 import compute_cv2
from pred_engine.ingesta.categorizacion.enrutador import route_syntetos_boylan
from pred_engine.ingesta.categorizacion.errores import (
    TopologyContractError,
    TopologyMathError,
    TopologyRoutingError,
)
from pred_engine.ingesta.categorizacion.panel import TopologyArtifact, classify_panel
from pred_engine.ingesta.categorizacion.proyeccion import select_canonical_columns

__all__ = [
    "TopologyArtifact",
    "TopologyContractError",
    "TopologyMathError",
    "TopologyRoutingError",
    "classify_panel",
    "compute_adi",
    "compute_cv2",
    "route_syntetos_boylan",
    "select_canonical_columns",
]
