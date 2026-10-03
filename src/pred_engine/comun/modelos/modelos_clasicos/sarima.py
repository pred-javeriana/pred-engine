"""Envoltorio SARIMAX determinista para el motor de HPO del Modulo 2."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

from pred_engine.comun.modelos.modelos_clasicos.errores import AjusteModeloError
from pred_engine.forecasting.base import BaseForecaster

# ADR-020: un ajuste con alguna raiz AR o MA a menos de 1.01 del origen esta en
# el borde de la region estacionaria o invertible y se rechaza, como hace
# auto.arima (paquete forecast, Hyndman y Khandakar, 2008).
MODULO_MINIMO_RAIZ = 1.01


def modulo_minimo_de_raices(ar: np.ndarray, ma: np.ndarray) -> float:
    """Menor modulo entre las raices de los polinomios AR y MA (orden creciente)."""
    modulos = [
        float(np.min(np.abs(np.roots(polinomio[::-1]))))
        for polinomio in (np.asarray(ar, dtype=float), np.asarray(ma, dtype=float))
        if polinomio.size > 1 and np.any(polinomio[1:] != 0.0)
    ]
    return min(modulos, default=float("inf"))


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
        self._resultado_ajuste: SARIMAXResultsWrapper | None = None

    def fit(self, y: np.ndarray) -> SarimaForecaster:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        serie = self._validar_serie_1d(y)
        self._fitted = False
        self._resultado_ajuste = None

        modelo = SARIMAX(
            serie,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.tendencia,
            # ADR-020: la estimacion se restringe a la region estacionaria e
            # invertible. Sin restriccion, un optimizador que no converge puede
            # devolver raices dentro del circulo unitario y un pronostico
            # explosivo (~1e63) que la media recortada del walk-forward oculta.
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resultado = cast(
                    "SARIMAXResultsWrapper",
                    modelo.fit(disp=False, maxiter=self.max_iter, return_params=False),
                )
        except (np.linalg.LinAlgError, ValueError) as exc:
            raise AjusteModeloError(
                f"SARIMA{self.order}x{self.seasonal_order} no convergio: {exc}"
            ) from exc

        modulo = modulo_minimo_de_raices(
            resultado.polynomial_ar, resultado.polynomial_ma
        )
        if modulo < MODULO_MINIMO_RAIZ:
            raise AjusteModeloError(
                f"SARIMA{self.order}x{self.seasonal_order} con una raiz AR/MA de "
                f"modulo {modulo:.4f} (< {MODULO_MINIMO_RAIZ}): ajuste en el borde "
                "de la region estacionaria o invertible"
            )
        self._resultado_ajuste = resultado
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
