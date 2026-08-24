"""Politica de almacenamiento inmutable y arbol de directorios de ingesta."""

from pred_engine.ingesta.data.layout import (
    DataLayout,
    default_data_root,
    ensure_data_layout,
)
from pred_engine.ingesta.data.proteccion_raw import (
    RawWritePermissionError,
    enforce_raw_read_only,
    raw_read_only_guard,
)

__all__ = [
    "DataLayout",
    "RawWritePermissionError",
    "default_data_root",
    "enforce_raw_read_only",
    "ensure_data_layout",
    "raw_read_only_guard",
]
