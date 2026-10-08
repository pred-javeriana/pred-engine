"""Identidad y estados de una corrida de M3 (ADR-03-010)."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Solo avanzan: EN_CURSO -> EVALUADA -> SELECCIONADA -> PUBLICADA. RECHAZADA se
# asigna al registrar un lote rechazado (ADR-03-005). Las dos ultimas son finales.
EstadoCorrida = Literal[
    "EN_CURSO", "EVALUADA", "SELECCIONADA", "PUBLICADA", "RECHAZADA"
]


class IdentidadCorrida(BaseModel):
    """Tupla de 3.0-A1 mas las versiones de politica y metricas.

    Los nombres son los de TASK-REC-3.0-A1; cuando exista su tipo, este se
    reemplaza por aquel. No incluye hash del contenido de los datos.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    ingesta_ref_m1: str = Field(min_length=1)
    configuracion_id: str = Field(min_length=1)
    codigo_version: str = Field(min_length=1)
    t_corte_reserva: date
    version_politica: str = Field(min_length=1)
    version_metricas: str = Field(min_length=1)

    def canonica(self) -> str:
        """JSON con claves ordenadas: la misma identidad da el mismo texto."""
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )

    @property
    def run_id(self) -> str:
        """Misma identidad, misma corrida; otra configuracion, corrida nueva."""
        digest = hashlib.sha256(self.canonica().encode("utf-8")).hexdigest()
        return "m3-" + digest[:16]
