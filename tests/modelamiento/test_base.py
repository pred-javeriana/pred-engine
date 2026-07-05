"""Tests for BaseForecaster contract and SeasonalNaiveStub."""

import numpy as np
import pytest

from pred_engine.modelamiento.base import BaseForecaster, SeasonalNaiveStub

# ---------------------------------------------------------------------------
# Contract tests via the concrete stub
# ---------------------------------------------------------------------------


def make_series(n: int = 21, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(10, 100, size=n)


class TestSeasonalNaiveStubFit:
    def test_returns_self(self):
        m = SeasonalNaiveStub(season_length=7)
        y = make_series()
        assert m.fit(y) is m

    def test_fitted_flag_set(self):
        m = SeasonalNaiveStub(season_length=7)
        assert not m._fitted
        m.fit(make_series())
        assert m._fitted

    def test_rejects_2d_array(self):
        m = SeasonalNaiveStub(season_length=7)
        with pytest.raises(ValueError, match="1-D"):
            m.fit(np.ones((3, 3)))

    def test_rejects_too_short_series(self):
        m = SeasonalNaiveStub(season_length=7)
        with pytest.raises(ValueError, match="at least"):
            m.fit(np.ones(3))

    def test_invalid_season_length(self):
        with pytest.raises(ValueError, match="season_length"):
            SeasonalNaiveStub(season_length=0)


class TestSeasonalNaiveStubPredict:
    def _fitted(self, season_length: int = 7) -> SeasonalNaiveStub:
        m = SeasonalNaiveStub(season_length=season_length)
        m.fit(make_series(n=season_length * 3))
        return m

    def test_output_length_equals_horizon(self):
        m = self._fitted()
        assert len(m.predict(14)) == 14

    def test_output_length_non_multiple(self):
        m = self._fitted(season_length=7)
        assert len(m.predict(10)) == 10

    def test_repeats_last_season(self):
        season_length = 4
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        m = SeasonalNaiveStub(season_length=season_length).fit(y)
        forecast = m.predict(4)
        np.testing.assert_array_equal(forecast, y[-season_length:])

    def test_predict_before_fit_raises(self):
        m = SeasonalNaiveStub(season_length=7)
        with pytest.raises(RuntimeError, match="fit"):
            m.predict(7)

    def test_predict_zero_horizon_raises(self):
        m = self._fitted()
        with pytest.raises(ValueError, match="horizon"):
            m.predict(0)

    def test_determinism_same_seed(self):
        y = make_series()
        f1 = SeasonalNaiveStub(seed=0).fit(y).predict(14)
        f2 = SeasonalNaiveStub(seed=0).fit(y).predict(14)
        np.testing.assert_array_equal(f1, f2)

    def test_output_is_numpy_array(self):
        m = self._fitted()
        result = m.predict(7)
        assert isinstance(result, np.ndarray)


class TestBaseForecasterInterface:
    def test_is_abstract(self):
        """BaseForecaster cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseForecaster()  # type: ignore[abstract]

    def test_seed_stored(self):
        m = SeasonalNaiveStub(seed=123)
        assert m.seed == 123
