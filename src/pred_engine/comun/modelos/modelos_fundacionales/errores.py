"""Errores de los modelos fundacionales."""

from __future__ import annotations

from pred_engine.comun.modelos.modelos_clasicos.errores import AjusteModeloError


class ModeloFundacionalNoDisponibleError(RuntimeError):
    """Falta el extra `foundation`, los pesos o la version fijada de la libreria."""


__all__ = ["AjusteModeloError", "ModeloFundacionalNoDisponibleError"]
