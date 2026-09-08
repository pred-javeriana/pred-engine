"""Envoltorio SARIMAX determinista para el motor de HPO del Modulo 2."""

from __future__ import annotations

import warnings

import numpy as np

from pred_engine.comun.modelos.modelos_clasicos.errores import AjusteModeloError
from pred_engine.forecasting.base import BaseForecaster


class SarimaForecaster(BaseForecaster):
    def __init__(
        self,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int],
        *,
        tendencia: str | None = None,
        seed: int = 0,
        max_iter: int = 50,
        forzar_no_negativo: bool = True,
    ) -> None:
        super().__init__(seed=seed, forzar_no_negativo=forzar_no_negativo)
        self.order = order
        self.seasonal_order = seasonal_order
        self.tendencia = tendencia
        self.max_iter = max_iter
        self._resultado_ajuste = None

    def fit(self, y: np.ndarray) -> SarimaForecaster:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        serie = self._validar_serie_1d(y)

        modelo = SARIMAX(
            serie,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.tendencia,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._resultado_ajuste = modelo.fit(disp=False, maxiter=self.max_iter)
        except (np.linalg.LinAlgError, ValueError) as exc:
            raise AjusteModeloError(
                f"SARIMA{self.order}x{self.seasonal_order} no convergio: {exc}"
            ) from exc

        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._require_fitted()
        self._validar_horizonte(horizon)
        assert self._resultado_ajuste is not None
        pronostico = np.asarray(
            self._resultado_ajuste.forecast(steps=horizon), dtype=float
        )
        return self._recortar_no_negativo(pronostico)
