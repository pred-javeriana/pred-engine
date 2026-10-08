"""Politica predeclarada y artefactos de seleccion y veredicto (3.4-A2).

Los veredictos son descriptivos (ADR-03-008): `VALIDADO` significa evidencia
descriptiva suficiente y favorable, no significancia estadistica. La prueba
HLN-DM acompana al veredicto como evidencia aparte y no cambia sus reglas.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from pred_engine.comun.modelos import SKU_CLASSES, SkuClass
from pred_engine.forecasting.evaluaciones.calculo_errores import Metricas
from pred_engine.forecasting.evaluaciones.diebold_mariano import PruebaDM

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


# ADR-03-003: familias elegibles por categoria, declaradas antes de abrir la
# reserva. Valor inicial: la matriz de enrutamiento 2.2 de M2.
FAMILIAS_ELEGIBLES_INICIALES: Mapping[SkuClass, tuple[str, ...]] = MappingProxyType(
    {
        "smooth": ("classical", "ml", "dl", "foundation"),
        "erratic": ("classical", "ml", "dl", "foundation"),
        "intermittent": ("classical", "ml", "foundation"),
        "lumpy": ("classical", "foundation"),
    }
)


@dataclass(frozen=True, slots=True)
class PoliticaSeleccion:
    """Se congela y versiona antes de abrir la reserva (ADR-03-003/007/008)."""

    version: str = "3.4.1-inicial"
    n_min: int = 30
    tolerancia_empate: float = 0.01
    orden_simplicidad: tuple[str, ...] = ("classical", "ml", "dl", "foundation")
    familias_elegibles: Mapping[SkuClass, tuple[str, ...]] = field(
        default_factory=lambda: FAMILIAS_ELEGIBLES_INICIALES
    )
    # Nivel de la prueba HLN-DM, con BH entre SKUs (ADR-03-008, alt. 2).
    alfa_dm: float = 0.05

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("la politica debe declarar su version")
        if self.n_min < 1:
            raise ValueError("n_min debe ser >= 1")
        if not self.tolerancia_empate >= 0:
            raise ValueError("tolerancia_empate debe ser >= 0")
        if not 0 < self.alfa_dm < 1:
            raise ValueError("alfa_dm debe estar en (0, 1)")
        if len(set(self.orden_simplicidad)) != len(self.orden_simplicidad):
            raise ValueError("orden_simplicidad no puede repetir familias")
        if set(self.familias_elegibles) != set(SKU_CLASSES):
            raise ValueError("familias_elegibles debe cubrir las cuatro categorias")
        for familias in self.familias_elegibles.values():
            if not familias or len(set(familias)) != len(familias):
                raise ValueError("cada categoria necesita familias sin repetir")


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
    # Familia -> "no_entregada" (elegible, sin candidatos validados de M2) o
    # "no_elegible" (entregada, fuera de la politica de la categoria).
    familias_excluidas: Mapping[str, str]


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
    diebold_mariano: PruebaDM
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
    # Origen de los datos con que se emitio; 3.5 lo contrasta con el de la corrida.
    datos_sinteticos: bool
    categorias: tuple[ResumenCategoria, ...]
    skus: tuple[VeredictoSku, ...]
