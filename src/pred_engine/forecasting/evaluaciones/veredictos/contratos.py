"""Politica predeclarada y artefactos de seleccion y veredicto (3.4-A2).

Los veredictos son descriptivos (ADR-03-008): `VALIDADO` significa evidencia
descriptiva suficiente y favorable, no significancia estadistica.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pred_engine.comun.modelos import SkuClass
from pred_engine.forecasting.evaluaciones.calculo_errores import Metricas

Veredicto = Literal[
    "FALLO_TECNICO",
    "NO_EVALUABLE",
    "EVIDENCIA_INSUFICIENTE",
    "VALIDADO",
    "EXPLORATORIO",
]
VEREDICTOS: tuple[Veredicto, ...] = (
    "FALLO_TECNICO",
    "NO_EVALUABLE",
    "EVIDENCIA_INSUFICIENTE",
    "VALIDADO",
    "EXPLORATORIO",
)

MotivoSeleccion = Literal[
    "menor_mediana",
    "empate_por_simplicidad",
    "compuerta_linea_base",
    "sin_skus_comparables",
]


@dataclass(frozen=True, slots=True)
class PoliticaSeleccion:
    """Se congela y versiona antes de abrir la reserva (ADR-03-007/008)."""

    version: str = "3.4.0-inicial"
    n_min: int = 30
    tolerancia_empate: float = 0.01
    orden_simplicidad: tuple[str, ...] = ("classical", "ml", "dl", "foundation")

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("la politica debe declarar su version")
        if self.n_min < 1:
            raise ValueError("n_min debe ser >= 1")
        if not self.tolerancia_empate >= 0:
            raise ValueError("tolerancia_empate debe ser >= 0")
        if len(set(self.orden_simplicidad)) != len(self.orden_simplicidad):
            raise ValueError("orden_simplicidad no puede repetir familias")


POLITICA_INICIAL = PoliticaSeleccion()


@dataclass(frozen=True, slots=True)
class SeleccionCategoria:
    """Familia campeona de una categoria; cada SKU usa su propia instancia."""

    sku_class: SkuClass
    familia_campeona: str
    medianas_r: Mapping[str, float]
    adverso: bool
    motivo: MotivoSeleccion
    skus_comparables: tuple[str, ...]
    excluidos: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ModeloEvaluado:
    """Dato descriptivo: un modelo evaluado en el SKU y su r frente a SN."""

    candidato_id: str
    familia: str
    modelo: str
    razon_sn: float | None


@dataclass(frozen=True, slots=True)
class VeredictoSku:
    sku: str
    sku_class: SkuClass
    veredicto: Veredicto
    familia_campeona: str
    candidato_campeon: str | None
    n_ventanas: int
    cobertura: float
    razon_sn: float | None
    pierde_frente_a_linea_base: bool | None
    comparacion_incompleta: bool
    metricas_campeon: Metricas | None
    metricas_linea_base: Metricas | None
    iqr_diferencia_mae: float | None
    justificacion: str
    modelos_evaluados: tuple[ModeloEvaluado, ...]


@dataclass(frozen=True, slots=True)
class ResumenCategoria:
    sku_class: SkuClass
    seleccion: SeleccionCategoria
    conteos: Mapping[Veredicto, int]
    porcentajes: Mapping[Veredicto, float]
    mediana_r: float | None
    n_adversos: int


@dataclass(frozen=True, slots=True)
class ResultadoEvaluacion:
    """Artefacto consolidado de 3.4; 3.5 lo persiste y M4 lo presenta."""

    version_politica: str
    version_metricas: str
    categorias: tuple[ResumenCategoria, ...]
    skus: tuple[VeredictoSku, ...]
