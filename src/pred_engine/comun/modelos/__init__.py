"""Contratos Pydantic compartidos por ingesta (y, mas adelante, otros modulos)."""

from pred_engine.comun.modelos.contrato import (
    CANONICAL_FIELDS,
    DiagnosticEntry,
    HeaderDiagnostic,
    InventoryObservation,
)

__all__ = [
    "CANONICAL_FIELDS",
    "DiagnosticEntry",
    "HeaderDiagnostic",
    "InventoryObservation",
]
