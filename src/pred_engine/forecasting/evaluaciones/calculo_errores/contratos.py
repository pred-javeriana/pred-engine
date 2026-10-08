"""Entrada y salida tipadas del calculo de metricas sobre la reserva (3.4-A1).

La entrada es minima a proposito: un pronostico fechado por origen. Hoy el
pipeline entrega un solo origen (t*, toda la reserva); cuando 3.3 genere las
ventanas de la reserva, cada ventana es un `PronosticoFechado` mas. La linea
base no llega en la entrada: 3.4 la genera en esas mismas ventanas.
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
    sku: str
    sku_class: SkuClass
    linea_base: MetricasCandidato
    candidatos: tuple[MetricasCandidato, ...]
    n_obs_validas_reserva: int
    reserva_toda_cero: bool
    n_candidatos_fallidos: int
