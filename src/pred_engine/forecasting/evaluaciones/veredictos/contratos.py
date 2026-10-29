"""Politica predeclarada y artefactos de seleccion y veredicto (3.4-A2, ADR-023).

La familia campeona de cada categoria se elige y se congela en el bloque de
seleccion de la reserva. Los veredictos, la prueba HLN-DM y la confirmacion
del campeon se calculan en el bloque de prueba, que ninguna decision lee.
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
from pred_engine.forecasting.evaluaciones.calculo_errores import (
    BloquesReserva,
    Metricas,
)
from pred_engine.forecasting.evaluaciones.confirmacion import (
    ConfirmacionCategoria,
    ParametrosConfirmacion,
)
from pred_engine.forecasting.evaluaciones.diebold_mariano import (
    IntervaloRazon,
    PruebaDM,
)
from pred_engine.optimizacion.router.politica import FAMILIES_BY_SKU_CLASS

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
FAMILIAS_ELEGIBLES_INICIALES: Mapping[SkuClass, tuple[str, ...]] = FAMILIES_BY_SKU_CLASS
# Chronos-2 se compara tambien con la mejor familia no fundacional (ADR-023).
FAMILIA_FUNDACIONAL = "foundation"


@dataclass(frozen=True, slots=True)
class PoliticaSeleccion:
    """Se congela y versiona antes de abrir la reserva (ADR-03-003/007/008, 023)."""

    version: str = "3.4.4-bloques"
    n_min: int = 30
    tolerancia_empate: float = 0.01
    orden_simplicidad: tuple[str, ...] = ("classical", "ml", "dl", "foundation")
    familias_elegibles: Mapping[SkuClass, tuple[str, ...]] = field(
        default_factory=lambda: FAMILIAS_ELEGIBLES_INICIALES
    )
    # Nivel de toda la inferencia (HLN-DM, BH, Holm e intervalos) y minimo de
    # origenes comunes de una prueba por SKU (ADR-03-011).
    alfa_dm: float = 0.05
    n_min_dm: int = 30
    # Parte de la reserva que elige al campeon; el resto es el bloque de prueba.
    fraccion_seleccion: float = 0.4
    # Confirmacion del campeon: SKU minimos por competidora y bootstrap de dias.
    n_min_confirmacion: int = 10
    bloque_bootstrap: int = 14
    replicas_bootstrap: int = 2000
    semilla_bootstrap: int = 20261029

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("la politica debe declarar su version")
        if type(self.n_min) is not int or self.n_min < 1:
            raise ValueError("n_min debe ser un entero >= 1")
        if not self.tolerancia_empate >= 0:
            raise ValueError("tolerancia_empate debe ser >= 0")
        if type(self.n_min_dm) is not int or self.n_min_dm < 2:
            raise ValueError("n_min_dm debe ser un entero >= 2")
        if not 0 < self.alfa_dm < 1:
            raise ValueError("alfa_dm debe estar en (0, 1)")
        if not 0 < self.fraccion_seleccion < 1:
            raise ValueError("fraccion_seleccion debe estar en (0, 1)")
        for nombre in ("n_min_confirmacion", "bloque_bootstrap", "replicas_bootstrap"):
            valor = getattr(self, nombre)
            if type(valor) is not int or valor < 1:
                raise ValueError(f"{nombre} debe ser un entero >= 1")
        if type(self.semilla_bootstrap) is not int or self.semilla_bootstrap < 0:
            raise ValueError("semilla_bootstrap debe ser un entero >= 0")
        if len(set(self.orden_simplicidad)) != len(self.orden_simplicidad):
            raise ValueError("orden_simplicidad no puede repetir familias")
        if set(self.familias_elegibles) != set(SKU_CLASSES):
            raise ValueError("familias_elegibles debe cubrir las cuatro categorias")
        for familias in self.familias_elegibles.values():
            if not familias or len(set(familias)) != len(familias):
                raise ValueError("cada categoria necesita familias sin repetir")
        object.__setattr__(self, "orden_simplicidad", tuple(self.orden_simplicidad))
        object.__setattr__(
            self,
            "familias_elegibles",
            MappingProxyType(
                {
                    clase: tuple(familias)
                    for clase, familias in self.familias_elegibles.items()
                }
            ),
        )

    @property
    def confirmacion(self) -> ParametrosConfirmacion:
        return ParametrosConfirmacion(
            alfa=self.alfa_dm,
            n_min_ventanas=self.n_min_dm,
            n_min_skus=self.n_min_confirmacion,
            bloque_dias=self.bloque_bootstrap,
            replicas=self.replicas_bootstrap,
            semilla=self.semilla_bootstrap,
        )


POLITICA_INICIAL = PoliticaSeleccion()


@dataclass(frozen=True, slots=True)
class SeleccionCategoria:
    """Familia campeona de una categoria; cada SKU usa su propia instancia.

    En el bloque de seleccion, o en los pliegues L3 previos a t* cuando es la
    seleccion previa de robustez (ADR-023).
    """

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
    """Veredicto y evidencia del SKU en el bloque de prueba (ADR-023).

    `intervalo_r` invierte HLN-DM al nivel 1 - alfa. `intervalo_r_fcr` existe
    solo para los SKU que BH selecciona, al nivel 1 - R q / m de su categoria
    (Benjamini y Yekutieli, 2005).
    """

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
    intervalo_r: IntervaloRazon | None = None
    intervalo_r_fcr: IntervaloRazon | None = None


@dataclass(frozen=True, slots=True)
class SignificanciaCategoria:
    """HLN-DM del campeon contra SN por SKU, con BH dentro de la categoria.

    `nivel_fcr` = 1 - R q / m: nivel de los intervalos de los R SKU que BH
    selecciona entre los m probados; `None` sin selecciones. Los conteos BY
    (Benjamini-Yekutieli) son la sensibilidad valida bajo cualquier dependencia.
    """

    n_probados: int
    n_mejores: int
    n_peores: int
    n_mejores_by: int
    n_peores_by: int
    nivel_fcr: float | None


@dataclass(frozen=True, slots=True)
class ResumenCategoria:
    """Seleccion, resumen de veredictos y confirmacion de una categoria (ADR-023).

    `contendiente` es la familia que la misma regla de seleccion elige sin la
    familia fundacional: la mejor familia no Chronos. `seleccion_previa` aplica
    la regla a los pliegues L3 previos a t*, como control de robustez.
    """

    sku_class: SkuClass
    seleccion: SeleccionCategoria
    conteos: Mapping[Veredicto, int]
    porcentajes: Mapping[Veredicto, float]
    mediana_r: float | None
    n_adversos: int
    significancia: SignificanciaCategoria
    confirmacion: ConfirmacionCategoria
    contendiente: str | None
    seleccion_previa: SeleccionCategoria | None


@dataclass(frozen=True, slots=True)
class ResultadoEvaluacion:
    """Artefacto consolidado de 3.4; 3.5 lo persiste y M4 lo presenta."""

    version_politica: str
    version_metricas: str
    # Origen de los datos con que se emitio; 3.5 lo contrasta con el de la corrida.
    datos_sinteticos: bool | None
    categorias: tuple[ResumenCategoria, ...]
    skus: tuple[VeredictoSku, ...]
    bloques: BloquesReserva
