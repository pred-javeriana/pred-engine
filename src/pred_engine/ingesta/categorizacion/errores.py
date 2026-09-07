"""Errores tipados del motor de topologia (fail-fast, sin PII)."""

from __future__ import annotations


class TopologyMathError(ValueError):
    """No se pudo calcular ADI o CV² (p. ej. division por cero)."""


class TopologyRoutingError(ValueError):
    """ADI/CV² no son enrutables (no finitos o negativos)."""


class TopologyContractError(ValueError):
    """El panel no cumple el contrato de entrada de la topologia."""
