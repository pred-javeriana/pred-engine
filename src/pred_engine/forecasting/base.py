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

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
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
