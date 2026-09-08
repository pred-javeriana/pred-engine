"""BaseForecaster contract and a trivial concrete stub for testing."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

import numpy as np


class BaseForecaster(ABC):
    """Abstract base class every PRED forecaster must implement.

    All models share a deterministic *seed* so that identical (data, config,
    seed) triples always produce identical forecasts (RNF-REP-01/02).
    """

    def __init__(self, seed: int = 0, *, forzar_no_negativo: bool = True) -> None:
        self.seed = seed
        self.forzar_no_negativo = forzar_no_negativo
        self._fitted = False

    @abstractmethod
    def fit(self, y: np.ndarray) -> BaseForecaster:
        """Fit the model on a 1-D demand series *y* (chronological order).

        Must return *self* so that ``m.fit(y).predict(h)`` is valid.
        """

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Return a 1-D forecast array of length *horizon*.

        Raises ``RuntimeError`` if called before :meth:`fit`.
        """

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

    def _seed_rng(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)

    @staticmethod
    def _validar_serie_1d(y: np.ndarray, nombre: str = "y") -> np.ndarray:
        """Convierte a float y exige 1D.

        Comun a todo `fit()` de un Pronosticador concreto (SARIMA, ML, DL,
        fundacional): evita repetir la misma validacion en cada subclase.
        """
        serie = np.asarray(y, dtype=float)
        if serie.ndim != 1:
            raise ValueError(f"{nombre} debe ser un array 1D")
        return serie

    @staticmethod
    def _validar_horizonte(horizon: int) -> None:
        """Exige horizon >= 1. Comun a todo `predict()` de un Pronosticador."""
        if horizon < 1:
            raise ValueError("horizon debe ser >= 1")

    def _recortar_no_negativo(self, pronostico: np.ndarray) -> np.ndarray:
        """Aplica la ley de conservacion de inventario si `forzar_no_negativo`.

        La demanda pronosticada no puede ser negativa; cada subclase decide
        si construye el pronostico a partir de esto (SARIMA, fundacional) en
        vez de repetir el mismo `np.clip` en cada `predict()`.
        """
        if self.forzar_no_negativo:
            return np.clip(pronostico, 0.0, None)
        return pronostico


class SeasonalNaiveStub(BaseForecaster):
    """Seasonal-naive model: last observed seasonal cycle repeated forward.

    This stub exists solely to exercise the ``BaseForecaster`` contract in
    the test suite.  It is *not* the production Seasonal Naive model.
    """

    def __init__(self, season_length: int = 7, seed: int = 0) -> None:
        """Initialize the seasonal-naive model.

        Args:
            season_length: Length of the seasonal cycle (must be >= 1).
            seed: Random seed for reproducibility.
        """
        super().__init__(seed=seed)
        if season_length < 1:
            raise ValueError("season_length must be >= 1")
        self.season_length = season_length
        self._last_season: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> SeasonalNaiveStub:
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be a 1-D array")
        if len(y) < self.season_length:
            raise ValueError(f"y must have at least {self.season_length} observations")
        self._last_season = y[-self.season_length :]
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._require_fitted()
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        assert self._last_season is not None
        reps = -(-horizon // self.season_length)  # ceiling division
        return np.tile(self._last_season, reps)[:horizon]
