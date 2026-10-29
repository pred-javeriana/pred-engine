"""Entrada y salida tipadas del calculo de metricas sobre la reserva (3.4-A1).

La entrada es minima a proposito: un pronostico fechado por cada origen diario
de la reserva (3.3). La linea base no llega en la entrada: 3.4 la genera en
esas mismas ventanas.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pred_engine.comun.modelos import SkuClass

VERSION_METRICAS = "3.4.1"

# La linea base no es una familia de M2: se identifica aparte (ADR-03-004).
FAMILIA_LINEA_BASE = "seasonal_naive"
MODELO_LINEA_BASE = "seasonal_naive"


@dataclass(frozen=True, slots=True, eq=False)
class PronosticoFechado:
    """Pronostico emitido en `origen` para las `fechas` de la reserva."""

    origen: pd.Timestamp
    fechas: pd.DatetimeIndex
    valores: np.ndarray


@dataclass(frozen=True, slots=True)
class SerieCandidato:
    """Todos los pronosticos de un candidato (o de la linea base) para un SKU."""

    candidato_id: str
    familia: str
    modelo: str
    pronosticos: tuple[PronosticoFechado, ...]


@dataclass(frozen=True, slots=True, eq=False)
class EntradaSku:
    """Evidencia de un SKU: historia <= t*, reserva real y pronosticos."""

    sku: str
    sku_class: SkuClass
    historia: np.ndarray
    reserva: pd.Series
    candidatos: tuple[SerieCandidato, ...]
    # Candidatos que no llegaron a pronosticar (3.2 o 3.3); cuentan para
    # `comparacion_incompleta` y `FALLO_TECNICO` (ADR-03-005).
    n_candidatos_fallidos: int = 0


@dataclass(frozen=True, slots=True)
class Metricas:
    """ADR-03-006, con e = y - y*. `None` solo si no es calculable."""

    n_pares: int
    mae: float
    rmse: float
    me: float
    mase: float | None
    razon_sn: float | None
    no_calculables: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class MetricasVentana:
    origen: pd.Timestamp
    metricas: Metricas


@dataclass(frozen=True, slots=True)
class MetricasCandidato:
    candidato_id: str
    familia: str
    modelo: str
    agregadas: Metricas | None
    por_ventana: tuple[MetricasVentana, ...]
    n_ventanas_totales: int
    n_ventanas_validas: int
    n_pronosticos_validos: int

    @property
    def cobertura(self) -> float:
        if self.n_ventanas_totales == 0:
            return 0.0
        return self.n_ventanas_validas / self.n_ventanas_totales


@dataclass(frozen=True, slots=True)
class EvaluacionSku:
    """Metricas de un SKU en un tramo de la reserva: toda ella o uno de sus bloques.

    Las observaciones validas y la demanda en cero se cuentan en los dias del tramo.
    """

    sku: str
    sku_class: SkuClass
    linea_base: MetricasCandidato
    candidatos: tuple[MetricasCandidato, ...]
    n_obs_validas_reserva: int
    reserva_toda_cero: bool
    n_candidatos_fallidos: int


@dataclass(frozen=True, slots=True)
class EvaluacionBloques:
    """Un SKU en toda la reserva y en cada bloque (ADR-023).

    Las tres vistas comparten las metricas por ventana; cambian los agregados y
    los conteos. La seleccion lee solo `seleccion`; veredictos, pruebas y efectos
    leen solo `prueba`; la publicacion por ventana lee `reserva`.
    """

    reserva: EvaluacionSku
    seleccion: EvaluacionSku
    prueba: EvaluacionSku

    @property
    def sku(self) -> str:
        return self.reserva.sku

    @property
    def sku_class(self) -> SkuClass:
        return self.reserva.sku_class
