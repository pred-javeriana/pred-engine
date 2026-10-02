"""Errores tipados del adaptador de candidatos M2 -> M3 (fail-closed, sin PII)."""

from __future__ import annotations

from pred_engine.forecasting.adaptador_candidatos.contratos import MotivoRechazo


class AdaptadorCandidatosError(Exception):
    """Frontera 3.2: el handoff de M2 no puede adaptarse para M3."""


class LoteRechazadoError(AdaptadorCandidatosError):
    """Rechazo fail-closed del lote completo: la reserva no se abre (ADR-03-005)."""

    def __init__(self, motivo: MotivoRechazo, detalle: str) -> None:
        self.motivo: MotivoRechazo = motivo
        self.detalle = detalle
        super().__init__(f"lote rechazado ({motivo}): {detalle}")


class ModeloNoRegistradoError(AdaptadorCandidatosError):
    """Se pidio instanciar una familia sin fabrica registrada."""
