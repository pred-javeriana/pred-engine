"""Pronosticadores de juguete deterministas para probar el motor sin statsmodels.

No empieza con `test_`: pytest no lo recolecta como modulo de pruebas, solo
se importa desde los `test_*.py` que lo necesitan.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class PronosticadorUltimoValor:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._ultimo: float | None = None

    def fit(self, y: np.ndarray) -> PronosticadorUltimoValor:
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            raise ValueError("y vacio")
        self._ultimo = float(y[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._ultimo is None:
            raise RuntimeError("llamar fit() antes de predict()")
        return np.full(horizon, self._ultimo, dtype=float)


def fabrica_ultimo_valor(
    configuracion: Mapping[str, Any], *, seed: int
) -> PronosticadorUltimoValor:
    return PronosticadorUltimoValor(seed=seed)


class PronosticadorEspia:
    def __init__(self, registro: list[int], seed: int = 0) -> None:
        self._registro = registro
        self.seed = seed
        self._ultimo = 0.0

    def fit(self, y: np.ndarray) -> PronosticadorEspia:
        y = np.asarray(y, dtype=float)
        self._registro.append(len(y))
        self._ultimo = float(y[-1]) if y.size else 0.0
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._ultimo, dtype=float)


def fabrica_espia(registro: list[int]):
    def _fabrica(configuracion: Mapping[str, Any], *, seed: int) -> PronosticadorEspia:
        return PronosticadorEspia(registro, seed=seed)

    return _fabrica


class PronosticadorNivelConstante:
    def __init__(self, nivel: float, seed: int = 0) -> None:
        self.nivel = nivel
        self.seed = seed
        self._fitted = False

    def fit(self, y: np.ndarray) -> PronosticadorNivelConstante:
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("llamar fit() antes de predict()")
        return np.full(horizon, self.nivel, dtype=float)


def fabrica_nivel_constante(
    configuracion: Mapping[str, Any], *, seed: int
) -> PronosticadorNivelConstante:
    return PronosticadorNivelConstante(nivel=configuracion["nivel"], seed=seed)


class PronosticadorRoto:
    def fit(self, y: np.ndarray) -> PronosticadorRoto:
        raise RuntimeError("fit deliberadamente roto")

    def predict(self, horizon: int) -> np.ndarray:
        raise RuntimeError("no deberia llamarse: fit ya fallo")


def fabrica_rota(configuracion: Mapping[str, Any], *, seed: int) -> PronosticadorRoto:
    return PronosticadorRoto()


class _ModeloRotoUnaVez:
    def fit(self, y: np.ndarray):
        raise RuntimeError("fit simulado: datos insuficientes en esta ventana")

    def predict(self, horizon: int) -> np.ndarray:
        raise RuntimeError("no deberia llamarse: fit ya fallo")


def fabrica_falla_primeras_n(n_fallos: int):
    contador = {"llamadas": 0}

    def _fabrica(configuracion: Mapping[str, Any], *, seed: int):
        contador["llamadas"] += 1
        if contador["llamadas"] <= n_fallos:
            return _ModeloRotoUnaVez()
        return PronosticadorUltimoValor(seed=seed)

    return _fabrica
