"""Artefactos de la particion causal y del recorrido Walk-Forward.

`ResultadoVentana`, `ResultadoWalkForward` y `EstadoParcial` usan
``eq=False``: contienen (o referencian, anidados) arrays de NumPy, y el
``__eq__`` que ``dataclass`` generaria por defecto compararia esos arrays
elemento a elemento dentro de una tupla, lo que revienta con
``ValueError: the truth value of an array is ambiguous``. Nunca comparar
estas instancias con ``==``; comparar campos escalares o arrays via
``numpy.testing.assert_array_equal`` explicitamente.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class VentanaTemporal:
    indice: int
    inicio_train: int
    fin_train: int
    inicio_val: int
    fin_val: int

    def __post_init__(self) -> None:
        if self.inicio_val != self.fin_train:
            raise ValueError(
                "inicio_val debe ser igual a fin_train (sin hueco ni solape)"
            )
        if self.fin_val <= self.inicio_val:
            raise ValueError("fin_val debe ser mayor que inicio_val")
        if self.fin_train <= self.inicio_train:
            raise ValueError("fin_train debe ser mayor que inicio_train")

    @property
    def horizonte(self) -> int:
        return self.fin_val - self.inicio_val

    @property
    def n_train(self) -> int:
        return self.fin_train - self.inicio_train


@dataclass(frozen=True, slots=True, eq=False)
class ResultadoVentana:
    ventana: VentanaTemporal
    metricas: Mapping[str, float]
    y_train: np.ndarray
    y_real: np.ndarray
    y_pred: np.ndarray
    duracion_s: float
    fallo: str | None = None


@dataclass(frozen=True, slots=True, eq=False)
class ResultadoWalkForward:
    ventanas: tuple[ResultadoVentana, ...]
    metrica_objetivo: str
    valor_agregado: float
    completo: bool
    n_ventanas_evaluadas: int
    n_ventanas_totales: int


@dataclass(frozen=True, slots=True, eq=False)
class EstadoParcial:
    ultima: ResultadoVentana
    n_evaluadas: int
    n_totales: int
    valor_parcial: float
