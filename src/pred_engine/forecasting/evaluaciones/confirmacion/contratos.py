"""Confirmacion del campeon de cada categoria en el bloque de prueba (ADR-023).

La familia campeona se congela en el bloque de seleccion. Aqui solo se mide en
el bloque de prueba si gana a cada familia competidora de su categoria; una
confirmacion fallida se informa y nunca cambia al campeon.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pred_engine.comun.modelos import SkuClass
from pred_engine.forecasting.evaluaciones.calculo_errores import MetricasCandidato

EstadoConfirmacion = Literal[
    "confirmado", "no_confirmado", "sin_comparables", "no_aplica"
]
ESTADOS_CONFIRMACION: tuple[EstadoConfirmacion, ...] = (
    "confirmado",
    "no_confirmado",
    "sin_comparables",
    "no_aplica",
)


@dataclass(frozen=True, slots=True)
class ParametrosConfirmacion:
    """Constantes predeclaradas; las fija la politica de seleccion."""

    alfa: float
    n_min_ventanas: int
    n_min_skus: int
    bloque_dias: int
    replicas: int
    semilla: int


@dataclass(frozen=True, slots=True)
class Intervalo:
    """Cotas de una razon; 0 e infinito son bordes validos."""

    inferior: float
    superior: float


@dataclass(frozen=True, slots=True)
class ComparacionFamilia:
    """Campeon frente a una competidora en los SKU con ambas instancias.

    La razon de cada SKU es RMSE_campeon / RMSE_competidora en sus ventanas
    comunes de prueba; < 1 favorece al campeon. Vale 0 si solo el campeon no
    tiene error, infinito si solo la competidora no lo tiene y 1 si ninguno lo
    tiene. ``p_iut`` es el p de Wilcoxon si la mediana favorece al campeon y 1 si
    no. ``n_mejores`` y ``n_peores`` cuentan los SKU con HLN-DM significativa por
    BH dentro de la comparacion. ``causa`` dice ``pocos_skus:<n><<minimo>`` sin
    prueba y ``perdida_cero:<campos>`` con los valores que quedan sin definir.
    """

    competidora: str
    n_skus: int
    mediana_razon: float | None
    ic_mediana: Intervalo | None
    hodges_lehmann: float | None
    ic_hodges_lehmann: Intervalo | None
    n_victorias: int
    ic_victorias: Intervalo | None
    estadistico_wilcoxon: float | None
    p_wilcoxon: float | None
    p_iut: float | None
    n_mejores: int | None
    n_peores: int | None
    causa: str | None


@dataclass(frozen=True, slots=True)
class EfectoLineaBase:
    """Mediana de r del campeon frente a Seasonal Naive en la categoria.

    ``causa`` es ``perdida_cero:<campos>`` si las perdidas cero dejan la mediana
    o su intervalo sin definir.
    """

    n_skus: int
    mediana_r: float | None
    ic_mediana: Intervalo | None
    causa: str | None = None


@dataclass(frozen=True, slots=True)
class ConfirmacionCategoria:
    sku_class: SkuClass
    familia_campeona: str
    estado: EstadoConfirmacion
    causa: str | None
    p_iut: float | None
    p_holm: float | None
    alfa: float
    comparaciones: tuple[ComparacionFamilia, ...]
    efecto_linea_base: EfectoLineaBase | None


@dataclass(frozen=True, slots=True, eq=False)
class CategoriaAConfirmar:
    """Lo que la confirmacion lee de una categoria: solo el bloque de prueba.

    ``instancias`` asigna a cada SKU sus instancias por familia, la linea base
    incluida, con las metricas de su bloque de prueba.
    """

    sku_class: SkuClass
    familia_campeona: str
    competidoras: tuple[str, ...]
    instancias: Mapping[str, Mapping[str, MetricasCandidato]]
