"""Politica de almacenamiento inmutable y arbol de directorios de ingesta."""

from pred_engine.ingesta.data.layout import (
    DataLayout,
    default_data_root,
    ensure_data_layout,
)

__all__ = [
    "DataLayout",
    "default_data_root",
    "ensure_data_layout",
]
