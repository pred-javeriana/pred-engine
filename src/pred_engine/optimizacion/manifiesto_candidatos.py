"""Manifiesto tipado de candidatos para el handoff M2 -> M3 (ADR-03-004).

M2 lo emite y M3 importa este mismo esquema para validarlo antes de abrir la
reserva: una union discriminada por ``familia``, con ``extra="forbid"`` y todos
los hiperparametros obligatorios. Ningun campo de configuracion tiene valor por
defecto, porque un hiperparametro ausente debe ser un error y no el default
silencioso de la libreria.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from pred_engine.comun.modelos import SkuClass
from pred_engine.comun.modelos.modelos_fundacionales.configuracion import (
    CHRONOS2_ZERO_SHOT,
)
from pred_engine.optimizacion.router.contratos import (
    SelectionResult,
    TopologicalProfile,
)

SCHEMA_VERSION = "1.0"


class _Estricto(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class ConfiguracionSarima(_Estricto):
    p: int = Field(ge=0)
    d: int = Field(ge=0)
    q: int = Field(ge=0)
    P: int = Field(ge=0)
    D: int = Field(ge=0)
    Q: int = Field(ge=0)
    m: int = Field(ge=0)
    tendencia: Literal["c", "n"]


class ConfiguracionLightGBM(_Estricto):
    lags: int = Field(ge=1)
    m: int = Field(ge=1)
    n_estimators: int = Field(ge=1)
    max_depth: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    min_child_weight: float = Field(ge=0)
    subsample: float = Field(gt=0, le=1)
    colsample_bytree: float = Field(gt=0, le=1)
    reg_alpha: float = Field(ge=0)
    reg_lambda: float = Field(ge=0)


class ConfiguracionMLP(_Estricto):
    lags: int = Field(ge=1)
    capas: int = Field(ge=2)
    unidades: int = Field(ge=1)
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0, le=1)
    dropout: float = Field(ge=0, lt=1)
    l2: float = Field(ge=0)


class ConfiguracionChronos2(_Estricto):
    """Configuracion congelada zero-shot, con la revision fija de los pesos."""

    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    version_libreria: str = Field(min_length=1)
    device: Literal["cpu"]
    dtype: Literal["float32"]
    cuantil_puntual: float = Field(gt=0, lt=1)
    aprendizaje_cruzado: bool
    max_contexto: int = Field(ge=1)
    n_hilos: int = Field(ge=1)


class ReferenciaCorrida(_Estricto):
    """Trazabilidad hacia M2; M3 no reutiliza veredictos de poda."""

    run_id: str = Field(min_length=1)
    estudio_hpo: str | None
    politica: str = Field(min_length=1)


class EvidenciaHPO(_Estricto):
    metrica_objetivo: str
    valor: float
    n_ventanas: int = Field(ge=0)
    n_trials: int = Field(ge=0)
    n_completados: int = Field(ge=0)
    n_podados: int = Field(ge=0)
    n_fallidos: int = Field(ge=0)


class _CandidatoBase(_Estricto):
    candidato_id: str = Field(min_length=1)
    sku: str = Field(min_length=1)
    sku_class: SkuClass
    perfil: TopologicalProfile
    corrida: ReferenciaCorrida
    semilla: int
    # None solo para la familia fundacional, que no ejecuta HPO.
    evidencia_hpo: EvidenciaHPO | None


class CandidatoClasico(_CandidatoBase):
    familia: Literal["classical"]
    modelo: Literal["sarima"]
    configuracion: ConfiguracionSarima
    estadistico_puntual: Literal["media"]


class CandidatoML(_CandidatoBase):
    familia: Literal["ml"]
    modelo: Literal["lightgbm"]
    configuracion: ConfiguracionLightGBM
    estadistico_puntual: Literal["media"]


class CandidatoDL(_CandidatoBase):
    familia: Literal["dl"]
    modelo: Literal["mlp"]
    configuracion: ConfiguracionMLP
    estadistico_puntual: Literal["media"]


class CandidatoFundacional(_CandidatoBase):
    familia: Literal["foundation"]
    modelo: Literal["chronos-2"]
    configuracion: ConfiguracionChronos2
    # Mediana de los cuantiles, fijada antes de abrir la reserva (Gneiting, 2011).
    estadistico_puntual: Literal["mediana"]


Candidato = Annotated[
    CandidatoClasico | CandidatoML | CandidatoDL | CandidatoFundacional,
    Field(discriminator="familia"),
]


class ContextoCorte(_Estricto):
    """Corte cronologico unificado de ADR-03-003."""

    t_estrella: date
    primer_dia_reservado: date
    ultimo_dia_observado: date
    dias_reservados: int = Field(ge=1)
    fraccion_reservada: float = Field(gt=0, lt=1)


class ManifiestoCandidatos(_Estricto):
    schema_version: Literal["1.0"]
    run_id: str = Field(min_length=1)
    contexto: ContextoCorte
    versiones: dict[str, str]
    candidatos: tuple[Candidato, ...]

    @model_validator(mode="after")
    def identidades_coherentes(self) -> ManifiestoCandidatos:
        identificadores = [c.candidato_id for c in self.candidatos]
        if len(set(identificadores)) != len(identificadores):
            raise ValueError("candidato_id repetido en el manifiesto")
        if any(c.corrida.run_id != self.run_id for c in self.candidatos):
            raise ValueError("todos los candidatos deben referir la corrida del run_id")
        return self


_MODELO_POR_FAMILIA = {
    "classical": "sarima",
    "ml": "lightgbm",
    "dl": "mlp",
    "foundation": "chronos-2",
}
_CANDIDATO = TypeAdapter(Candidato)


def candidato_desde_seleccion(seleccion: SelectionResult, *, run_id: str) -> Candidato:
    """Traduce un `SelectionResult` al esquema validado, sin completar defaults."""
    if seleccion.forecast_config is None:
        raise ValueError(f"{seleccion.sku_id}/{seleccion.family} sin forecast_config")
    payload = seleccion.payload
    fundacional = seleccion.family == "foundation"
    configuracion: Mapping[str, Any] = (
        CHRONOS2_ZERO_SHOT.descripcion_canonica()
        if fundacional
        else seleccion.forecast_config
    )
    evidencia = None
    if not fundacional:
        evidencia = {
            campo: payload[campo]
            for campo in (
                "metrica_objetivo",
                "valor",
                "n_ventanas",
                "n_trials",
                "n_completados",
                "n_podados",
                "n_fallidos",
            )
        }
    return _CANDIDATO.validate_python(
        {
            "candidato_id": f"{seleccion.family}:{seleccion.sku_id}",
            "sku": seleccion.sku_id,
            "sku_class": seleccion.sku_class,
            "perfil": seleccion.profile,
            "familia": seleccion.family,
            "modelo": _MODELO_POR_FAMILIA[seleccion.family],
            "configuracion": dict(configuracion),
            "estadistico_puntual": "mediana" if fundacional else "media",
            "semilla": seleccion.forecast_seed,
            "corrida": {
                "run_id": run_id,
                "estudio_hpo": payload.get("estudio_hpo"),
                "politica": seleccion.policy_version or "sin-version",
            },
            "evidencia_hpo": evidencia,
        }
    )


def construir_manifiesto(
    selecciones: Sequence[SelectionResult],
    *,
    run_id: str,
    contexto: ContextoCorte,
    versiones: Mapping[str, str],
) -> ManifiestoCandidatos:
    return ManifiestoCandidatos(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        contexto=contexto,
        versiones=dict(versiones),
        candidatos=tuple(
            candidato_desde_seleccion(s, run_id=run_id) for s in selecciones
        ),
    )
