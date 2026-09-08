"""Contrato canonico de una observacion de demanda PRED y de su topologia."""

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

# Columna inyectada por 1.3. No forma parte de InventoryObservation.
TOPOLOGY_FIELD = "sku_class"
PANEL_FIELDS: tuple[str, ...] = CANONICAL_FIELDS + (TOPOLOGY_FIELD,)

# Tipos pandas del artefacto 1.4 (unico origen de verdad para normalizar).
PANEL_DTYPES: dict[str, str] = {
    "sku_id": "string",
    "timestamp": "datetime64[ns]",
    "demand_qty": "float64",
    "lead_time_days": "int64",
    "sku_class": "string",
}

# Syntetos, Boylan & Croston (2005); Johnston & Boylan (1996).
ADI_THRESHOLD: float = 1.32
CV2_THRESHOLD: float = 0.49

SkuClass = Literal["smooth", "intermittent", "erratic", "lumpy"]
SKU_CLASSES: tuple[SkuClass, ...] = (
    "smooth",
    "intermittent",
    "erratic",
    "lumpy",
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


class ClassifiedObservation(InventoryObservation):
    """Fila del contrato final 1.4: observacion diaria mas sku_class constante."""

    sku_class: SkuClass


class TopologyMetrics(BaseModel):
    """Metricas de un SKU: se calculan una vez y se reutilizan en todas las filas."""

    model_config = ConfigDict(strict=True, extra="forbid")

    sku_id: str = Field(min_length=1)
    n_periods: int = Field(ge=1)
    n_positive: int = Field(ge=0)
    adi: float = Field(gt=0)
    cv2: float = Field(ge=0)
    sku_class: SkuClass

    @field_validator("sku_id")
    @classmethod
    def sku_topologia_sin_espacios(cls, valor: str) -> str:
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
