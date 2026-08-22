"""Errores de continuidad temporal."""

from __future__ import annotations


class TemporalContinuityError(ValueError):
    """No se pudo construir una cuadrícula diaria valida."""
