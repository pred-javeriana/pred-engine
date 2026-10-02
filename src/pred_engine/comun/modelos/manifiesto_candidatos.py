"""Manifiesto tipado del handoff M2 -> M3 (ADR-03-004).

Una sola definicion: M2 lo produce (`optimizacion.router.manifiesto`) y M3 lo
valida (`forecasting.adaptador_candidatos`). Cada familia tiene su esquema con
todos los hiperparametros obligatorios: un campo ausente nunca se completa con
el default de la libreria, porque M3 evaluaria otra configuracion sin notarlo.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Annotated, Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)

from pred_engine.comun.modelos.contrato import SkuClass
from pred_engine.comun.modelos.modelos_fundacionales.configuracion import (
    CHRONOS2_ZERO_SHOT,
)

VERSION_ESQUEMA_MANIFIESTO = 1

# Nombre del modelo concreto de cada familia del hito (el catalogo no crece).
MODELO_POR_FAMILIA: dict[str, str] = {
    "classical": "sarima",
    "ml": "lightgbm",
    "dl": "mlp",
    "foundation": "chronos2",
}

_ESTRICTO = ConfigDict(strict=True, extra="forbid", frozen=True, allow_inf_nan=False)
_CHRONOS2_FIJADO = CHRONOS2_ZERO_SHOT.descripcion_canonica()


def configuracion_declarada(
    familia: str, forecast_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Configuracion que M2 declara en el manifiesto para una familia.

    La familia fundacional no se configura (su `forecast_config` es vacio):
    declara la configuracion congelada, asi M3 verifica la revision de pesos.
    """
    if familia == "foundation":
        return dict(_CHRONOS2_FIJADO)
    return dict(forecast_config)


class ContextoParticion(BaseModel):
    """Particion de 3.1 con la que M2 optimizo; M3 la exige identica."""

    model_config = _ESTRICTO

    ingesta_ref_m1: str = Field(min_length=1)
    t_corte_reserva: date
    fraccion_reserva: float = Field(gt=0.0, lt=1.0)


class ConfigSarima(BaseModel):
    """Claves que lee `fabrica_sarima`; `m=0` significa sin estacionalidad."""

    model_config = _ESTRICTO

    p: int = Field(ge=0)
    d: int = Field(ge=0)
    q: int = Field(ge=0)
    P: int = Field(ge=0)
    D: int = Field(ge=0)
    Q: int = Field(ge=0)
    m: int = Field(ge=0)

    @field_validator("m")
    @classmethod
    def estacionalidad_coherente(cls, valor: int, info: ValidationInfo) -> int:
        if valor == 1:
            raise ValueError("m debe ser 0 (sin estacionalidad) o >= 2")
        if valor == 0 and any(info.data.get(campo) for campo in ("P", "D", "Q")):
            raise ValueError("con m=0 la parte estacional (P, D, Q) debe ser 0")
        return valor


class ConfigLightGBM(BaseModel):
    """Claves que lee `fabrica_ml`; `m` es la estacionalidad de las features."""

    model_config = _ESTRICTO

    lags: int = Field(ge=1)
    m: int = Field(ge=1)
    n_estimators: int = Field(ge=1)
    max_depth: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    min_child_weight: float = Field(ge=0.0)
    subsample: float = Field(gt=0.0, le=1.0)
    colsample_bytree: float = Field(gt=0.0, le=1.0)
    reg_alpha: float = Field(ge=0.0)
    reg_lambda: float = Field(ge=0.0)


class ConfigMLP(BaseModel):
    """Argumentos de `MLPForecaster` (sin la semilla, que viaja aparte)."""

    model_config = _ESTRICTO

    lags: int = Field(ge=1)
    capas: int = Field(ge=2)
    unidades: int = Field(ge=1)
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0, le=1.0)
    dropout: float = Field(ge=0.0, lt=1.0)
    l2: float = Field(ge=0.0, le=1.0)


class ConfigChronos2(BaseModel):
    """Copia declarada de `CHRONOS2_ZERO_SHOT`: fija la revision de pesos y el
    estadistico puntual (cuantil 0.5) antes de abrir la reserva."""

    model_config = _ESTRICTO

    model_id: str
    revision: str
    version_libreria: str
    device: Literal["cpu"]
    dtype: Literal["float32"]
    cuantil_puntual: float
    aprendizaje_cruzado: bool
    max_contexto: int
    n_hilos: int

    @field_validator("*")
    @classmethod
    def igual_a_la_configuracion_fijada(cls, valor: Any, info: ValidationInfo) -> Any:
        fijado = _CHRONOS2_FIJADO[info.field_name or ""]
        if valor != fijado:
            raise ValueError(f"debe ser {fijado!r} (CHRONOS2_ZERO_SHOT)")
        return valor


class _CandidatoBase(BaseModel):
    model_config = _ESTRICTO

    candidato_id: str = Field(min_length=1)
    sku: str = Field(min_length=1)
    sku_class: SkuClass
    semilla: int = Field(ge=0)

    @field_validator("candidato_id", "sku")
    @classmethod
    def texto_sin_solo_espacios(cls, valor: str) -> str:
        limpio = valor.strip()
        if not limpio:
            raise ValueError("el campo de texto no puede ser vacio")
        return limpio


class CandidatoClasico(_CandidatoBase):
    familia: Literal["classical"]
    modelo: Literal["sarima"]
    configuracion: ConfigSarima


class CandidatoML(_CandidatoBase):
    familia: Literal["ml"]
    modelo: Literal["lightgbm"]
    configuracion: ConfigLightGBM


class CandidatoDL(_CandidatoBase):
    familia: Literal["dl"]
    modelo: Literal["mlp"]
    configuracion: ConfigMLP


class CandidatoFundacional(_CandidatoBase):
    familia: Literal["foundation"]
    modelo: Literal["chronos2"]
    configuracion: ConfigChronos2


Candidato = Annotated[
    CandidatoClasico | CandidatoML | CandidatoDL | CandidatoFundacional,
    Field(discriminator="familia"),
]


class ManifiestoCandidatos(BaseModel):
    """Sobre del handoff. Los candidatos viajan crudos: M3 los valida uno a uno
    para que un candidato mal formado no rechace el lote (ADR-03-005)."""

    model_config = _ESTRICTO

    schema_version: int
    run_id_m2: str = Field(min_length=1)
    emitido_en: AwareDatetime
    contexto: ContextoParticion
    candidatos: tuple[dict[str, Any], ...]
