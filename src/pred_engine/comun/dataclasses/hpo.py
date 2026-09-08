"""Clases del motor de HPO: `Trial` y `ResultadoEstudio`.

Compartidos por las tres familias (clasicos/ML/DL)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pred_engine.comun.dataclasses.validacion_temporal import VentanaTemporal

EstadoTrial = Literal["pendiente", "corriendo", "completado", "podado", "fallido"]


@dataclass(frozen=True, slots=True)
class Trial:
    id: str
    configuracion: Mapping[str, Any]
    estado: EstadoTrial
    valor: float | None
    n_ventanas: int
    motivo: str | None
    metrica_objetivo: str
    familia: str
    sku_id: str | None = None


@dataclass(frozen=True, slots=True, eq=False)
class ResultadoEstudio:
    mejor: Trial | None
    trials: tuple[Trial, ...]
    n_completados: int
    n_podados: int
    n_fallidos: int
    ventanas: tuple[VentanaTemporal, ...]
    metrica_objetivo: str
    seed: int
