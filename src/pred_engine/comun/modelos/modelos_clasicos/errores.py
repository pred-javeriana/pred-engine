"""Errores de ajuste de modelos clasicos."""

from __future__ import annotations

from pred_engine.comun.hechos_error import ErrorConHechos


class AjusteModeloError(ErrorConHechos, RuntimeError):
    """El modelo no se ajusto o no pronostico; ``hechos`` dice por que (``fit``)."""
