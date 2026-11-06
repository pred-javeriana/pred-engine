"""Clases del motor de HPO: `Trial` y `ResultadoEstudio`.

Compartidos por las tres familias (clasicos/ML/DL)
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pred_engine.comun.dataclasses.validacion_temporal import (
    ResultadoWalkForward,
    VentanaTemporal,
)

EstadoTrial = Literal["pendiente", "corriendo", "completado", "podado", "fallido"]


def causa_trial(motivo: str | None) -> str:
    """Categoria comun para contar descartes y elegir su primer motivo."""
    if not motivo:
        return "sin_motivo_registrado"
    # Los registros previos anteponen ventana/valor a la causa del descarte.
    prefijo = re.search(
        r"\b(configuracion_invalida|ajuste_fallido|poda_semantica|escalon)\b",
        motivo,
    )
    if prefijo is not None:
        return prefijo.group(1)
    if "asha" in motivo:
        return "escalon"
    return motivo.split(":", 1)[0]


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
        """Totales, conteos por causa y su primer motivo, sin volcar cada trial."""
        motivos: dict[str, str] = {}
        for trial in self.trials:
            if trial.estado in ("fallido", "podado"):
                motivos.setdefault(
                    causa_trial(trial.motivo),
                    trial.motivo or "sin_motivo_registrado",
                )
        return (
            f"{self.n_fallidos} fallidos, {self.n_podados} podados; "
            f"causas={dict(self.causas)}; motivos={motivos}"
        )
