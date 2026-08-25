"""Errores de alineacion semantica (fail-fast consultivo)."""

from __future__ import annotations

import json

from pred_engine.comun.modelos import HeaderDiagnostic


class SemanticAlignmentError(ValueError):
    """PRED no puede ingerir el dataset: la sonda rechazo o invalido el diagnostico."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: HeaderDiagnostic | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic

    def diagnostic_json(self) -> str:
        """Serializa el reporte para stdout o logs estructurados."""
        if self.diagnostic is None:
            return json.dumps(
                {"status": "rejected", "diagnostic": []},
                ensure_ascii=False,
            )
        return self.diagnostic.model_dump_json(ensure_ascii=False)
