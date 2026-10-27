"""Errores tipados del handoff 1.4 (fail-closed, sin PII)."""

from __future__ import annotations

from pred_engine.comun.errores_handoff import (
    MissingCutoffMetadataError,
    OutputHandoffError,
)

__all__ = [
    "HandoffPreconditionError",
    "MissingCutoffMetadataError",
    "OutputContractError",
    "OutputHandoffError",
    "PanelPreservationError",
]


class OutputContractError(OutputHandoffError):
    """El panel clasificado viola el contrato de cinco columnas."""


class PanelPreservationError(OutputHandoffError):
    """El clasificador altero filas o columnas transaccionales del panel diario."""


class HandoffPreconditionError(OutputHandoffError):
    """El panel diario no es clasificable (p. ej. SKU sin demanda positiva)."""
