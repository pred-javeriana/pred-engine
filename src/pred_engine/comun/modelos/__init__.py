"""Contratos Pydantic compartidos por ingesta (y, mas adelante, otros modulos)."""

from pred_engine.comun.modelos.contrato import (
    ADI_THRESHOLD,
    CANONICAL_FIELDS,
    CV2_THRESHOLD,
    PANEL_DTYPES,
    PANEL_FIELDS,
    SKU_CLASSES,
    TOPOLOGY_FIELD,
    ClassifiedObservation,
    DiagnosticEntry,
    HeaderDiagnostic,
    InventoryObservation,
    SkuClass,
    TopologyMetrics,
)

__all__ = [
    "ADI_THRESHOLD",
    "CANONICAL_FIELDS",
    "CV2_THRESHOLD",
    "PANEL_DTYPES",
    "PANEL_FIELDS",
    "SKU_CLASSES",
    "TOPOLOGY_FIELD",
    "ClassifiedObservation",
    "DiagnosticEntry",
    "HeaderDiagnostic",
    "InventoryObservation",
    "SkuClass",
    "TopologyMetrics",
]
