"""Errores de alineacion semantica (fail-closed)."""

from __future__ import annotations


class SemanticAlignmentError(ValueError):
    """PRED no puede trabajar con este dataset: el mapeo es incompleto o invalido."""
