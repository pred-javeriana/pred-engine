"""Errores del motor de HPO (agnostico de familia)."""

from __future__ import annotations


class EspacioInvalidoError(ValueError):
    pass


class EstudioError(RuntimeError):
    pass
