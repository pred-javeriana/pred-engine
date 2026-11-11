"""Errores del motor de HPO (agnostico de familia)."""

from __future__ import annotations

from pred_engine.comun.hechos_error import ErrorConHechos


class EspacioInvalidoError(ValueError):
    pass


class EstudioError(ErrorConHechos, RuntimeError):
    """El estudio no pudo completarse; ``hechos`` describe la busqueda agotada."""
