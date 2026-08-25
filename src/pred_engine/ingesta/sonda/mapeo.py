"""Artefacto de la sonda consultiva: marco intacto + diagnostico JSON."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pred_engine.comun.modelos import HeaderDiagnostic


@dataclass(frozen=True, slots=True)
class DiagnosticArtifact:
    """Marco sin mutar mas el reporte de la sonda LLM."""

    frame: pd.DataFrame
    diagnostic: HeaderDiagnostic
