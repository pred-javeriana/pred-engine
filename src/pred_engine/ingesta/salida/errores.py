"""Errores tipados del handoff 1.4 (fail-closed, sin PII)."""

from __future__ import annotations


class OutputHandoffError(ValueError):
    """Frontera 1.4: el panel no puede publicarse hacia los modulos 2 y 3."""


class OutputContractError(OutputHandoffError):
    """El panel clasificado viola el contrato de cinco columnas."""


class PanelPreservationError(OutputHandoffError):
    """El clasificador altero filas o columnas transaccionales del panel diario."""


class HandoffPreconditionError(OutputHandoffError):
    """El panel diario no es clasificable (p. ej. SKU sin demanda positiva)."""
