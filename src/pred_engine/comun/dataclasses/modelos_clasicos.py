"""Artefactos de salida de la seleccion clasica.

Ver optimizacion.optimizadores.modelos_clasicos.classical_selection.

REESCRITO: la version anterior de este archivo definia `ConfiguracionEvaluada`
(con un campo `aicc`) y tenia un `NameError` de importacion (`field` sin
importar). Ambos problemas quedan resueltos aqui. La preseleccion por AICc
(ADR-02-002) esta depreciada; la seleccion vigente es HPO sobre Walk-Forward
Validation (ADR-02-009), que no produce un ranking por AICc sino un ganador
por metrica fuera de muestra.
"""

from __future__ import annotations

from dataclasses import dataclass

from pred_engine.comun.dataclasses.hpo import ResultadoEstudio, Trial


@dataclass(frozen=True, slots=True)
class ConfiguracionSeleccionada:
    sku_id: str | None
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    metrica_objetivo: str
    valor: float
    n_ventanas: int


@dataclass(frozen=True, slots=True, eq=False)
class ResultadoSeleccionClasica:
    seleccionada: ConfiguracionSeleccionada | None
    estudio: ResultadoEstudio

    @property
    def descartadas(self) -> tuple[Trial, ...]:
        return tuple(
            t for t in self.estudio.trials if t.estado in ("podado", "fallido")
        )
