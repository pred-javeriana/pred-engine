"""Contratos del ciclo de vida de una corrida PRED -- frontera con el backend HPO.

No importan Optuna ni Walk-Forward. El checkpoint del motor es una
referencia opaca (`backend_checkpoint`); quien la reconstruye es el
adaptador inyectado (paso B2).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = 1

BackendHPO = Literal["optuna"]


class EstadoCorrida(StrEnum):
    """Ciclo de vida de la corrida PRED, no de un trial de Optuna."""

    NUEVA = "nueva"
    EN_PROGRESO = "en_progreso"
    INTERRUMPIDA = "interrumpida"
    COMPLETADA = "completada"
    FALLIDA = "fallida"


class ConfiguracionValidacion(BaseModel):
    """Parametros de Walk-Forward que condicionan la reproducibilidad."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    min_train: int = Field(ge=1)
    horizonte: int = Field(ge=1)
    paso: int = Field(ge=1)
    estacionalidad: int = Field(ge=1)


class ConfiguracionOptimizador(BaseModel):
    """Recorte serializable de reglas de poda + presupuesto + muestreador."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    n_trials_objetivo: int = Field(ge=1)
    muestreador: str = Field(min_length=1)
    min_ventanas: int = Field(ge=1)
    agregacion: Literal["media", "mediana", "media_recortada"]
    proporcion_recorte: float = Field(ge=0.0, le=0.5)
    factor_reduccion: int = Field(ge=2)
    habilitar_poda_semantica: bool


class SolicitudCorrida(BaseModel):
    """Pedido de crear o reanudar. `run_id` identifica; no entra en la huella."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    familia: str = Field(min_length=1)
    sku_id: str = Field(min_length=1)
    seed: int
    metrica_objetivo: str = Field(min_length=1)
    validacion: ConfiguracionValidacion
    espacio_busqueda: dict[str, Any]
    optimizador: ConfiguracionOptimizador
    n_observaciones: int = Field(ge=1)
    huella_serie: str = Field(min_length=1)
    backend: BackendHPO = "optuna"

    @field_validator("run_id", "familia", "sku_id", "metrica_objetivo", "huella_serie")
    @classmethod
    def texto_sin_solo_espacios(cls, valor: str) -> str:
        limpio = valor.strip()
        if not limpio:
            raise ValueError("el campo de texto no puede ser vacio")
        return limpio


class ManifiestoCorrida(BaseModel):
    """Metadatos persistidos de la corrida. Independiente de Optuna."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    schema_version: int = Field(ge=1)
    run_id: str = Field(min_length=1)
    # strict=False: el JSON persiste el valor en minusculas, no el miembro enum.
    estado: EstadoCorrida = Field(strict=False)
    familia: str = Field(min_length=1)
    sku_id: str = Field(min_length=1)
    seed: int
    metrica_objetivo: str = Field(min_length=1)
    configuracion_validacion: ConfiguracionValidacion
    fingerprint_configuracion: str = Field(min_length=1)
    backend: BackendHPO
    backend_checkpoint: str | None = None
    n_trials_objetivo: int = Field(ge=1)
    n_trials_finalizados: int = Field(ge=0)

    @field_validator("run_id", "familia", "sku_id", "metrica_objetivo")
    @classmethod
    def texto_sin_solo_espacios(cls, valor: str) -> str:
        limpio = valor.strip()
        if not limpio:
            raise ValueError("el campo de texto no puede ser vacio")
        return limpio


@runtime_checkable
class AlmacenManifiestos(Protocol):
    """Puerto de persistencia del manifiesto (inyectable; paso B1)."""

    def guardar(self, manifiesto: ManifiestoCorrida) -> None: ...

    def cargar(self, run_id: str) -> ManifiestoCorrida: ...

    def existe(self, run_id: str) -> bool: ...


@runtime_checkable
class PuertoPersistenciaMotor(Protocol):
    """Puerto del backend HPO (inyectable; paso B2). `handle` es opaco."""

    def crear(self) -> object: ...

    def persistir(self, handle: object) -> str: ...

    def restaurar(self, referencia: str) -> object: ...
