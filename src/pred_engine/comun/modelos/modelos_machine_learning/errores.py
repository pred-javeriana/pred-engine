"""Errores de ajuste de modelos de Machine Learning."""

from __future__ import annotations

from pred_engine.comun.modelos.modelos_clasicos.errores import AjusteModeloError


class SerieCortaError(ValueError):
    """La serie no alcanza para construir ni una muestra de entrenamiento util."""


__all__ = ["AjusteModeloError", "SerieCortaError"]
