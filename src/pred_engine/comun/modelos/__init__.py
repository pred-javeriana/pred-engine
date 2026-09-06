"""Contratos Pydantic compartidos por ingesta (y, mas adelante, otros modulos)."""

from pred_engine.comun.modelos.contrato import (
    CANONICAL_FIELDS,
    HANDOFF_FIELDS,
    SKU_CLASS_LABELS,
    DiagnosticEntry,
    HeaderDiagnostic,
    InventoryObservation,
)

__all__ = [
    "CANONICAL_FIELDS",
    "HANDOFF_FIELDS",
    "SKU_CLASS_LABELS",
    "DiagnosticEntry",
    "HeaderDiagnostic",
    "InventoryObservation",
]
