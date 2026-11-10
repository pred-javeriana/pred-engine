"""Errores de ajuste de modelos de Machine Learning."""

from __future__ import annotations

from pred_engine.comun.hechos_error import ErrorConHechos
from pred_engine.comun.modelos.modelos_clasicos.errores import AjusteModeloError


class SerieCortaError(ErrorConHechos, ValueError):
    """La serie no alcanza para construir ni una muestra de entrenamiento util.

    ``hechos`` lleva ``history``: observaciones exigidas y disponibles.
    """


__all__ = ["AjusteModeloError", "SerieCortaError"]
