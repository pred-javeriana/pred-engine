"""MLP autoregresivo pequeno: tanh, salida lineal y SGD por mini-batches.

Dos o mas capas ocultas entrenables, sin dependencias adicionales ni estado
aleatorio global. Cada fit reinicia pesos y normalizacion usando solo el prefijo
que entrega Walk-Forward. No hay validacion aleatoria ni early stopping interno.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pred_engine.forecasting.base import BaseForecaster

MUESTRAS_MINIMAS = 5


def numero_parametros(lags: int, capas: int, unidades: int) -> int:
    return (
        (lags + 1) * unidades + (capas - 1) * (unidades + 1) * unidades + unidades + 1
    )


class MLPForecaster(BaseForecaster):
    def __init__(
        self,
        *,
        lags: int = 7,
        capas: int = 2,
        unidades: int = 16,
        epochs: int = 30,
        batch_size: int = 16,
        learning_rate: float = 0.01,
        dropout: float = 0.0,
        l2: float = 1e-4,
        seed: int = 0,
        forzar_no_negativo: bool = True,
    ) -> None:
        super().__init__(seed=seed, forzar_no_negativo=forzar_no_negativo)
        for nombre, valor, minimo in (
            ("lags", lags, 1),
            ("capas", capas, 2),
            ("unidades", unidades, 1),
            ("epochs", epochs, 1),
            ("batch_size", batch_size, 1),
        ):
            if type(valor) is not int or valor < minimo:
                raise ValueError(f"{nombre} debe ser un entero >= {minimo}")
        if not np.isfinite(learning_rate) or not 0 < learning_rate <= 1:
            raise ValueError("learning_rate debe estar en (0, 1]")
        if not np.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout debe estar en [0, 1)")
        if not np.isfinite(l2) or not 0 <= l2 <= 1:
            raise ValueError("l2 debe estar en [0, 1]")
        self.lags = lags
        self.capas = capas
        self.unidades = unidades
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.l2 = l2
        self._pesos: list[np.ndarray] = []
        self._sesgos: list[np.ndarray] = []
        self._historia: np.ndarray | None = None
        self._media = 0.0
        self._escala = 1.0

    def _gradientes(
        self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activaciones = [x]
        ocultas = []
        mascaras = []
        for peso, sesgo in zip(self._pesos[:-1], self._sesgos[:-1], strict=True):
            oculta = np.tanh(activaciones[-1] @ peso + sesgo)
            mascara = (
                (rng.random(oculta.shape) >= self.dropout) / (1 - self.dropout)
                if self.dropout
                else np.ones_like(oculta)
            )
            ocultas.append(oculta)
            mascaras.append(mascara)
            activaciones.append(oculta * mascara)
        pred = activaciones[-1] @ self._pesos[-1] + self._sesgos[-1]
        delta = 2 * (pred - y) / len(y)
        grad_pesos = []
        grad_sesgos = []
        for i in reversed(range(len(self._pesos))):
            grad_pesos.append(activaciones[i].T @ delta + self.l2 * self._pesos[i])
            grad_sesgos.append(delta.sum(axis=0))
            if i:
                delta = (
                    (delta @ self._pesos[i].T)
                    * mascaras[i - 1]
                    * (1 - ocultas[i - 1] ** 2)
                )
        return grad_pesos[::-1], grad_sesgos[::-1]

    def fit(self, y: np.ndarray) -> MLPForecaster:
        # Un refit fallido nunca deja utilizable el modelo de un prefijo anterior.
        self._fitted = False
        self._historia = None
        serie = self._validar_serie_1d(y)
        if not np.all(np.isfinite(serie)):
            raise ValueError("y contiene valores no finitos")
        if len(serie) < self.lags + MUESTRAS_MINIMAS:
            raise ValueError(
                f"historial insuficiente: se requieren lags + {MUESTRAS_MINIMAS} "
                "observaciones para entrenar"
            )
        rng = np.random.default_rng(self.seed)
        dimensiones = [self.lags, *([self.unidades] * self.capas), 1]
        self._pesos = [
            rng.normal(0, np.sqrt(2 / (entrada + salida)), (entrada, salida))
            for entrada, salida in zip(dimensiones[:-1], dimensiones[1:], strict=True)
        ]
        self._sesgos = [np.zeros(salida) for salida in dimensiones[1:]]
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                self._media = float(serie.mean())
                self._escala = max(float(serie.std()), 1e-8)
                normalizada = (serie - self._media) / self._escala
                muestras = sliding_window_view(normalizada, self.lags + 1)
                x, objetivos = muestras[:, :-1], muestras[:, -1:]
                for _ in range(self.epochs):
                    for inicio in range(0, len(x), self.batch_size):
                        fin = inicio + self.batch_size
                        gp, gs = self._gradientes(
                            x[inicio:fin], objetivos[inicio:fin], rng
                        )
                        for peso, sesgo, dp, ds in zip(
                            self._pesos, self._sesgos, gp, gs, strict=True
                        ):
                            peso -= self.learning_rate * dp
                            sesgo -= self.learning_rate * ds
        except FloatingPointError as exc:
            raise ValueError("entrenamiento DL no finito") from exc
        if not all(np.isfinite(p).all() for p in self._pesos + self._sesgos):
            raise ValueError("entrenamiento DL no finito")
        self._historia = normalizada[-self.lags :].copy()
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._require_fitted()
        if type(horizon) is not int:
            raise ValueError("horizon debe ser un entero")
        self._validar_horizonte(horizon)
        assert self._historia is not None
        historia = self._historia.copy()
        predicciones = np.empty(horizon, dtype=float)
        for paso in range(horizon):
            activacion = historia.reshape(1, -1)
            for peso, sesgo in zip(self._pesos[:-1], self._sesgos[:-1], strict=True):
                activacion = np.tanh(activacion @ peso + sesgo)
            valor = float((activacion @ self._pesos[-1] + self._sesgos[-1]).item())
            valor = valor * self._escala + self._media
            if not np.isfinite(valor):
                raise ValueError("pronostico DL no finito")
            if self.forzar_no_negativo:
                valor = max(0.0, valor)
            predicciones[paso] = valor
            historia = np.append(historia[1:], (valor - self._media) / self._escala)
        return predicciones


def fabrica_dl(configuracion: Mapping[str, Any], *, seed: int) -> MLPForecaster:
    """FabricaPronosticador: configuracion ganadora reutilizable sin conversiones."""
    return MLPForecaster(**dict(configuracion), seed=seed)
