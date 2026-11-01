"""Clases del motor de HPO: `Trial` y `ResultadoEstudio`.

Compartidos por las tres familias (clasicos/ML/DL)
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pred_engine.comun.dataclasses.validacion_temporal import (
    ResultadoWalkForward,
    VentanaTemporal,
)

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
    timestamp: str | None = None


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
    causas: Mapping[str, int] = field(default_factory=dict)
    # Walk-forward completo de ``mejor`` si este estudio lo evaluo (no si lo
    # reconstruyo de un manifiesto); es el mismo que L3 calcularia.
    walk_forward_mejor: ResultadoWalkForward | None = None

    @property
    def resumen_causas(self) -> str:
        """Conteos y motivos originales para explicar un estudio sin ganador."""
        motivos = Counter(
            t.motivo or "sin_motivo_registrado"
            for t in self.trials
            if t.estado in ("fallido", "podado")
        )
        return f"causas={dict(self.causas)}; motivos={dict(motivos)}"
