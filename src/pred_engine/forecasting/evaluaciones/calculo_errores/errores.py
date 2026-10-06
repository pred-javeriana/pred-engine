"""Errores tipados de la evaluacion retrospectiva (fail-closed, sin PII)."""

from __future__ import annotations


class EvaluacionRetrospectivaError(ValueError):
    """Frontera 3.4: la evidencia recibida no permite evaluar sobre la reserva."""
