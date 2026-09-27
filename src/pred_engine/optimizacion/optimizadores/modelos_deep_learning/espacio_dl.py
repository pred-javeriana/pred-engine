"""Arquitectura y entrenamiento del MLP, expresados en el DSL comun de HPO."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pred_engine.comun.modelos.modelos_deep_learning.mlp import (
    MUESTRAS_MINIMAS,
    MLPForecaster,
    numero_parametros,
)
from pred_engine.optimizacion.optimizadores.HPO.espacio import (
    Entero,
    EspacioBusqueda,
    Flotante,
)


@dataclass(frozen=True, slots=True)
class EspacioDL:
    lags: tuple[int, int] = (3, 14)
    capas: tuple[int, int] = (2, 3)
    unidades: tuple[int, int] = (8, 32)
    epochs: tuple[int, int] = (10, 50)
    batch_size: tuple[int, int] = (8, 32)
    learning_rate: tuple[float, float] = (0.001, 0.05)
    dropout: tuple[float, float] = (0.0, 0.3)
    l2: tuple[float, float] = (1e-8, 0.01)
    costo_max: int = 100_000

    def __post_init__(self) -> None:
        limites = {
            nombre: getattr(self, nombre)
            for nombre in (
                "lags",
                "capas",
                "unidades",
                "epochs",
                "batch_size",
                "learning_rate",
                "dropout",
                "l2",
            )
        }
        for nombre, rango in limites.items():
            if len(rango) != 2 or rango[0] > rango[1]:
                raise ValueError(f"rango invalido para {nombre}")
        # Ambos extremos deben ser configuraciones entrenables; no se asignan pesos.
        for extremo in (0, 1):
            MLPForecaster(
                **{nombre: rango[extremo] for nombre, rango in limites.items()}
            )
        if self.l2[0] <= 0:
            raise ValueError("l2 debe ser positivo para la busqueda logaritmica")
        if type(self.costo_max) is not int or self.costo_max < 1:
            raise ValueError("costo_max debe ser un entero positivo")
        minima = {nombre: rango[0] for nombre, rango in limites.items()}
        if not _CostoAcotado(self.costo_max)(minima):
            raise ValueError("costo_max excluye todas las configuraciones DL")


@dataclass(frozen=True, slots=True)
class _CostoAcotado:
    # El repr estable incluye la cota: el DSL comun la incorpora en su huella,
    # por lo que cambiarla no permite reanudar una busqueda incompatible.
    maximo: int

    def __call__(self, configuracion: Mapping[str, Any]) -> bool:
        parametros = numero_parametros(
            configuracion["lags"], configuracion["capas"], configuracion["unidades"]
        )
        return parametros * configuracion["epochs"] <= self.maximo


def construir_espacio_dl(espacio: EspacioDL | None = None) -> EspacioBusqueda:
    cfg = espacio or EspacioDL()
    return EspacioBusqueda(
        parametros=(
            Entero("lags", *cfg.lags),
            Entero("capas", *cfg.capas),
            Entero("unidades", *cfg.unidades),
            Entero("epochs", *cfg.epochs),
            Entero("batch_size", *cfg.batch_size),
            Flotante("learning_rate", *cfg.learning_rate, log=True),
            Flotante("dropout", *cfg.dropout),
            Flotante("l2", *cfg.l2, log=True),
        ),
        restricciones=(_CostoAcotado(cfg.costo_max),),
    )


def min_train_recomendado_dl(espacio: EspacioDL | None = None) -> int:
    return (espacio or EspacioDL()).lags[1] + MUESTRAS_MINIMAS
