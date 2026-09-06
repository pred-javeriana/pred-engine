"""Error del contrato de frontera 1.4 (fail-closed)."""

from __future__ import annotations


class HandoffContractError(ValueError):
    """El panel clasificado no puede publicarse hacia el modulo 2."""
