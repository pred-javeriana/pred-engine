"""Contrato canonico de una observacion de demanda PRED."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Unico origen de verdad para la sonda y la barrera.
CANONICAL_FIELDS: tuple[str, ...] = (
    "sku_id",
    "timestamp",
    "demand_qty",
    "lead_time_days",
)


class InventoryObservation(BaseModel):
    """Fila ya tipada que puede cruzar hacia el remuestreo y el modulo 2."""

    model_config = ConfigDict(strict=True, extra="forbid")

    sku_id: str = Field(min_length=1)
    timestamp: datetime
    demand_qty: float = Field(ge=0)
    lead_time_days: int = Field(ge=1)

    @field_validator("sku_id")
    @classmethod
    def sku_sin_solo_espacios(cls, valor: str) -> str:
        # El extractor conserva strings crudos; un SKU en blanco no es identificador.
        limpio = valor.strip()
        if not limpio:
            raise ValueError("sku_id no puede ser vacio")
        return limpio


class DiagnosticEntry(BaseModel):
    """Entrada del reporte JSON de la sonda consultiva."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    severity: Literal["error", "info"] = "error"
    message: str = Field(min_length=1)
    action: str | None = None

    @field_validator("action", mode="before")
    @classmethod
    def accion_vacia_a_nulo(cls, valor: object) -> object:
        if valor is None:
            return None
        if isinstance(valor, str):
            limpio = valor.strip()
            return limpio or None
        raise TypeError("action debe ser string o nulo")


class HeaderDiagnostic(BaseModel):
    """Payload JSON de la sonda: aceptado o rechazado con instrucciones."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["accepted", "rejected"]
    diagnostic: tuple[DiagnosticEntry, ...] = ()

    @field_validator("diagnostic", mode="before")
    @classmethod
    def normalizar_lista(cls, valor: object) -> object:
        if valor is None:
            return ()
        return valor

    def is_accepted(self) -> bool:
        return self.status == "accepted"

    def is_rejected(self) -> bool:
        return self.status == "rejected"
