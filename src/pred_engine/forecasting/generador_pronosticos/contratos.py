"""Protocolo y resultado del pronostico reservado (3.1 y 3.3)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import pandas as pd

from pred_engine.comun.modelos.manifiesto_candidatos import Candidato
from pred_engine.forecasting.adaptador_candidatos import PERIODO_LINEA_BASE
from pred_engine.forecasting.evaluaciones.calculo_errores import (
    PronosticoFechado,
    SerieCandidato,
)

CodigoExclusion = Literal[
    "INSUFFICIENT_HISTORY", "CALENDAR_GAPS", "INVALID_HISTORY", "INSUFFICIENT_RESERVE"
]
CodigoFalloPronostico = Literal[
    "ajuste_fallido",
    "pronostico_fallido",
    "pronostico_invalido",
    "observaciones_ausentes",
]


@dataclass(frozen=True, slots=True)
class ConfiguracionPronosticos:
    """Se fija antes de abrir la reserva; el paso entre origenes siempre es diario.

    ``horizonte`` no se acorta al llegar al final de la reserva. La historia
    minima es explicita y nunca menor que la necesaria para Seasonal Naive.
    ``codigo_version`` identifica la implementacion que produjo las unidades.
    """

    horizonte: int
    codigo_version: str
    min_historia: int = PERIODO_LINEA_BASE

    def __post_init__(self) -> None:
        if type(self.horizonte) is not int or self.horizonte < 1:
            raise ValueError("horizonte debe ser un entero positivo")
        if type(self.min_historia) is not int or self.min_historia < PERIODO_LINEA_BASE:
            raise ValueError(f"min_historia debe ser >= {PERIODO_LINEA_BASE}")
        if not self.codigo_version.strip():
            raise ValueError("codigo_version no puede estar vacia")


@dataclass(frozen=True, slots=True)
class VentanaReserva:
    origen: pd.Timestamp
    fechas: pd.DatetimeIndex


@dataclass(frozen=True, slots=True)
class FalloPronostico:
    """Mismo formato de fallo de 3.2, con el origen de la unidad afectada."""

    sku: str
    candidato_id: str
    modelo: str
    origen: pd.Timestamp
    campo: str
    codigo_error: CodigoFalloPronostico
    mensaje: str


@dataclass(frozen=True, slots=True, eq=False)
class UnidadPronostico:
    """Resultado completo que el coordinador confirma en una sola transaccion.

    Incluye configuracion, semilla, version y emision. Un fallo conserva las
    fechas planeadas con valores NaN, nunca ceros que parezcan un pronostico.
    """

    candidato: Candidato
    codigo_version: str
    emitido_en: datetime
    pronostico: PronosticoFechado
    fallo: FalloPronostico | None = None


@dataclass(frozen=True, slots=True, eq=False)
class ResultadoPronosticos:
    candidato: Candidato
    unidades: tuple[UnidadPronostico, ...]

    @property
    def serie(self) -> SerieCandidato:
        return SerieCandidato(
            candidato_id=self.candidato.candidato_id,
            familia=self.candidato.familia,
            modelo=self.candidato.modelo,
            pronosticos=tuple(u.pronostico for u in self.unidades),
        )

    @property
    def fallos(self) -> tuple[FalloPronostico, ...]:
        return tuple(u.fallo for u in self.unidades if u.fallo is not None)


class SerieNoElegibleError(ValueError):
    """Exclusion anterior al ajuste; no modifica el horizonte ni el corte."""

    def __init__(self, sku: str, codigo: CodigoExclusion, detalle: str) -> None:
        self.sku = sku
        self.codigo = codigo
        super().__init__(f"sku={sku} codigo={codigo}: {detalle}")
