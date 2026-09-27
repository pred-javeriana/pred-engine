"""Predictor DL entrenable compatible con el Walk-Forward comun."""

from pred_engine.comun.modelos.modelos_deep_learning.mlp import (
    MLPForecaster,
    fabrica_dl,
)

__all__ = ["MLPForecaster", "fabrica_dl"]
