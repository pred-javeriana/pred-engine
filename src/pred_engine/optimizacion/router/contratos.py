"""Contratos comunes del motor de seleccion -- frontera con ingesta y estrategias.

Reutiliza `SkuClass` y `ClassifiedObservation` del Modulo 1. No importa
Optuna, Walk-Forward, ni implementaciones concretas de modelos.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pred_engine.comun.modelos import ClassifiedObservation, SkuClass

PredictorFamily = Literal["classical", "ml", "dl", "foundation"]
PREDICTOR_FAMILIES: tuple[PredictorFamily, ...] = (
    "classical",
    "ml",
    "dl",
    "foundation",
)

TopologicalProfile = Literal[
    "dense_stable",
    "dense_variable",
    "sparse_stable",
    "sparse_variable",
]
TOPOLOGICAL_PROFILES: tuple[TopologicalProfile, ...] = (
    "dense_stable",
    "dense_variable",
    "sparse_stable",
    "sparse_variable",
)


class SelectionRequest(BaseModel):
    """Entrada tipada del motor. No lee archivos; la serie es opcional."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    sku_id: str = Field(min_length=1)
    sku_class: SkuClass
    series: tuple[ClassifiedObservation, ...] = ()

    @field_validator("sku_id")
    @classmethod
    def sku_sin_solo_espacios(cls, valor: str) -> str:
        # Misma regla que InventoryObservation: un SKU en blanco no identifica.
        limpio = valor.strip()
        if not limpio:
            raise ValueError("sku_id no puede ser vacio")
        return limpio

    @model_validator(mode="after")
    def serie_coherente_con_identidad(self) -> SelectionRequest:
        for indice, observacion in enumerate(self.series):
            if observacion.sku_id != self.sku_id:
                raise ValueError(
                    f"series[{indice}].sku_id no coincide con la solicitud"
                )
            if observacion.sku_class != self.sku_class:
                raise ValueError(
                    f"series[{indice}].sku_class no coincide con la solicitud"
                )
        return self


class SelectionResult(BaseModel):
    """Resultado comun de una estrategia. El payload es opaco para el enrutador."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    sku_id: str = Field(min_length=1)
    sku_class: SkuClass
    family: PredictorFamily
    profile: TopologicalProfile
    produced_by: str = Field(min_length=1)
    policy_version: str = ""
    payload: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("sku_id", "produced_by")
    @classmethod
    def texto_sin_solo_espacios(cls, valor: str) -> str:
        limpio = valor.strip()
        if not limpio:
            raise ValueError("el campo de texto no puede ser vacio")
        return limpio


@runtime_checkable
class SelectionStrategy(Protocol):
    """Contrato estructural que debe satisfacer cualquier estrategia registrada."""

    family: PredictorFamily

    def select(
        self, request: SelectionRequest, profile: TopologicalProfile
    ) -> SelectionResult: ...
