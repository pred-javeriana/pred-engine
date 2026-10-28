"""Errores base de handoff entre L1 y los modulos posteriores."""

from __future__ import annotations


class OutputHandoffError(ValueError):
    """Frontera 1.4: el panel no puede publicarse hacia los modulos 2 y 3."""


class MissingCutoffMetadataError(OutputHandoffError):
    """El Parquet no contiene los metadatos requeridos de corte y exclusiones."""
