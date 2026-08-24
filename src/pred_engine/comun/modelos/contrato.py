"""Contrato canonico de una observacion de demanda PRED."""

from __future__ import annotations

from datetime import datetime

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


class HeaderMapping(BaseModel):
    """Diccionario canonico → columna fuente. None = el LLM no pudo alinear."""

    model_config = ConfigDict(extra="ignore")

    sku_id: str | None = None
    timestamp: str | None = None
    demand_qty: str | None = None
    lead_time_days: str | None = None

    @field_validator("*", mode="before")
    @classmethod
    def vacio_a_nulo(cls, valor: object) -> object:
        if valor is None:
            return None
        if isinstance(valor, str):
            limpio = valor.strip()
            return limpio or None
        raise TypeError("cada campo del mapeo debe ser string o nulo")

    def unmapped_fields(self) -> tuple[str, ...]:
        faltantes = [
            campo for campo in CANONICAL_FIELDS if getattr(self, campo) is None
        ]
        return tuple(faltantes)

    def source_to_canonical(self) -> dict[str, str]:
        """Invertido para df.rename(columns=...). Omite nulos."""
        invertido: dict[str, str] = {}
        for canonico in CANONICAL_FIELDS:
            fuente = getattr(self, canonico)
            if fuente is None:
                continue
            if fuente in invertido and invertido[fuente] != canonico:
                raise ValueError(
                    "una columna fuente no puede mapear a dos campos canonicos: "
                    f"{fuente}"
                )
            invertido[fuente] = canonico
        return invertido
