"""Contrato estructural entre el motor de validacion y los modelos.

`comun.walkforward` NO importa `pred_engine.forecasting` ni ningun
motor de modelos concreto (statsmodels, etc.): habla con ellos a traves de
este `Protocol`. Esto permite (a) probar todo el Walk-Forward con un
pronosticador de juguete en milisegundos, y (b) que las familias ML y DL
reutilicen el mismo motor sin que `comun` conozca su existencia.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Pronosticador(Protocol):
    def fit(self, y: np.ndarray) -> Pronosticador: ...

    def predict(self, horizon: int) -> np.ndarray: ...


class FabricaPronosticador(Protocol):
    def __call__(
        self, configuracion: Mapping[str, Any], *, seed: int
    ) -> Pronosticador: ...
