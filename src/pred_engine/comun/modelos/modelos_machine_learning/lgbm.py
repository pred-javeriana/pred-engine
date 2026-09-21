"""Pronosticador LightGBM determinista para el motor de HPO del Modulo 2.

Regresion autoregresiva: cada muestra usa las ultimas `lags` observaciones
(mas tres agregados de esa misma ventana y, si hay estacionalidad, la fase
`posicion % m`) para predecir el siguiente valor. El pronostico a `horizon`
pasos es recursivo: cada prediccion se reinyecta como observacion.

Las features de una muestra usan SOLO observaciones anteriores al objetivo,
asi que no hay fuga temporal dentro de `fit`, y el Walk-Forward no puede
filtrar el futuro porque este modelo solo ve el `y` que se le pasa a `fit`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import lightgbm as lgb
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pred_engine.comun.modelos.modelos_machine_learning.errores import (
    AjusteModeloError,
    SerieCortaError,
)
from pred_engine.forecasting.base import BaseForecaster

# Muestras de entrenamiento minimas por encima de `lags` para que el ajuste
# tenga sentido; por debajo, un arbol solo memoriza.
MUESTRAS_MINIMAS = 5

# Con hessiano constante (error cuadratico) `min_child_weight` cuenta muestras;
# este piso evita que el default de LightGBM (20) impida todo split en series
# cortas de Walk-Forward.
_MIN_MUESTRAS_HOJA = 3
_HOJAS_MAX = 31


def _features(
    ventanas: np.ndarray, posiciones: np.ndarray, estacionalidad: int
) -> np.ndarray:
    """Matriz (n, lags + 3 [+ 1]) a partir de ventanas de lags ya alineadas."""
    columnas = [
        ventanas,
        ventanas.mean(axis=1, keepdims=True),
        ventanas.std(axis=1, keepdims=True),
        (ventanas == 0).mean(axis=1, keepdims=True),
    ]
    if estacionalidad > 1:
        columnas.append((posiciones % estacionalidad).reshape(-1, 1).astype(float))
    return np.hstack(columnas)


def construir_muestras(
    serie: np.ndarray, lags: int, estacionalidad: int
) -> tuple[np.ndarray, np.ndarray]:
    """(X, objetivos) autoregresivos. La fila i usa serie[i : i + lags] y
    predice serie[i + lags]: ninguna feature toca el objetivo ni el futuro."""
    ventanas = sliding_window_view(serie, lags)[:-1]
    posiciones = np.arange(lags, len(serie))
    return _features(ventanas, posiciones, estacionalidad), serie[lags:]


class LightGBMForecaster(BaseForecaster):
    def __init__(
        self,
        *,
        lags: int = 7,
        estacionalidad: int = 1,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        seed: int = 0,
        forzar_no_negativo: bool = True,
    ) -> None:
        super().__init__(seed=seed, forzar_no_negativo=forzar_no_negativo)
        if lags < 1:
            raise ValueError("lags debe ser >= 1")
        if estacionalidad < 1:
            raise ValueError("estacionalidad debe ser >= 1")
        self.lags = lags
        self.estacionalidad = estacionalidad
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self._modelo: lgb.Booster | None = None
        self._historia: np.ndarray | None = None

    def _parametros_lightgbm(self) -> dict[str, Any]:
        return {
            "objective": "regression",
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": min(_HOJAS_MAX, 2**self.max_depth),
            "min_data_in_leaf": _MIN_MUESTRAS_HOJA,
            "min_sum_hessian_in_leaf": self.min_child_weight,
            "bagging_fraction": self.subsample,
            "bagging_freq": 1 if self.subsample < 1.0 else 0,
            "feature_fraction": self.colsample_bytree,
            "lambda_l1": self.reg_alpha,
            "lambda_l2": self.reg_lambda,
            "seed": self.seed,
            # Determinismo: un hilo y particion de filas fija (ver ADR-014).
            "num_threads": 1,
            "deterministic": True,
            "force_row_wise": True,
            "verbosity": -1,
        }

    def fit(self, y: np.ndarray) -> LightGBMForecaster:
        serie = self._validar_serie_1d(y)
        if not np.all(np.isfinite(serie)):
            raise ValueError("y contiene valores no finitos")
        minimo = self.lags + MUESTRAS_MINIMAS
        if len(serie) < minimo:
            raise SerieCortaError(
                f"serie de longitud {len(serie)} insuficiente: con lags="
                f"{self.lags} se requieren al menos {minimo} observaciones"
            )

        X, objetivos = construir_muestras(serie, self.lags, self.estacionalidad)

        try:
            self._modelo = lgb.train(
                self._parametros_lightgbm(),
                lgb.Dataset(X, label=objetivos),
                num_boost_round=self.n_estimators,
            )
        except (lgb.basic.LightGBMError, ValueError) as exc:
            raise AjusteModeloError(f"LightGBM no ajusto: {exc}") from exc

        self._historia = serie.copy()
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._require_fitted()
        self._validar_horizonte(horizon)
        assert self._modelo is not None and self._historia is not None

        historia = self._historia
        pronostico = np.empty(horizon, dtype=float)
        for paso in range(horizon):
            ventana = historia[-self.lags :].reshape(1, -1)
            posicion = np.array([len(historia)])
            X = _features(ventana, posicion, self.estacionalidad)
            valor = float(np.asarray(self._modelo.predict(X)).ravel()[0])
            if self.forzar_no_negativo:
                valor = max(valor, 0.0)
            pronostico[paso] = valor
            historia = np.append(historia, valor)
        return self._recortar_no_negativo(pronostico)


def fabrica_ml(configuracion: Mapping[str, Any], *, seed: int) -> LightGBMForecaster:
    """`FabricaPronosticador` para el motor de HPO.

    `configuracion` es la salida del espacio de busqueda (ver
    `construir_espacio_ml`); las claves ausentes usan el default del
    forecaster. `m` (estacionalidad) no se busca: lo inyecta el llamador.
    """
    return LightGBMForecaster(
        lags=int(configuracion.get("lags", 7)),
        estacionalidad=int(configuracion.get("m", 1)),
        n_estimators=int(configuracion.get("n_estimators", 100)),
        max_depth=int(configuracion.get("max_depth", 4)),
        learning_rate=float(configuracion.get("learning_rate", 0.1)),
        min_child_weight=float(configuracion.get("min_child_weight", 1.0)),
        subsample=float(configuracion.get("subsample", 1.0)),
        colsample_bytree=float(configuracion.get("colsample_bytree", 1.0)),
        reg_alpha=float(configuracion.get("reg_alpha", 0.0)),
        reg_lambda=float(configuracion.get("reg_lambda", 0.0)),
        seed=seed,
    )
