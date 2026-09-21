"""Artefactos de salida de la seleccion de Machine Learning (2.5 / 6.0).

Analogos a `modelos_clasicos.py`: la seleccion es HPO sobre Walk-Forward, y el
ganador se decide por metrica fuera de muestra, no por un criterio de
informacion (AICc no aplica a modelos sin parametrizacion p,d,q).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pred_engine.comun.dataclasses.hpo import ResultadoEstudio, Trial


@dataclass(frozen=True, slots=True)
class ConfiguracionSeleccionadaML:
    sku_id: str | None
    hiperparametros: Mapping[str, Any]
    metrica_objetivo: str
    valor: float
    n_ventanas: int


@dataclass(frozen=True, slots=True, eq=False)
class ResultadoSeleccionML:
    seleccionada: ConfiguracionSeleccionadaML | None
    estudio: ResultadoEstudio

    @property
    def descartadas(self) -> tuple[Trial, ...]:
        return tuple(
            t for t in self.estudio.trials if t.estado in ("podado", "fallido")
        )
