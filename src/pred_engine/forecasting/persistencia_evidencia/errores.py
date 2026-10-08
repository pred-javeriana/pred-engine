"""Errores tipados de la persistencia de evidencia de M3 (sin PII)."""

from __future__ import annotations


class PersistenciaEvidenciaError(Exception):
    """Frontera 3.5: la evidencia no puede escribirse o publicarse."""


class EstadoCorridaError(PersistenciaEvidenciaError):
    """La operacion no se permite en el estado actual de la corrida (ADR-03-010).

    Cubre la corrida no registrada (`estado=None`) y las terminales: sobre una
    corrida PUBLICADA o RECHAZADA no se escribe; re-evaluar exige otro `run_id`.
    """

    def __init__(self, run_id: str, estado: str | None, operacion: str) -> None:
        self.run_id = run_id
        self.estado = estado
        self.operacion = operacion
        situacion = "no esta registrada" if estado is None else f"esta en {estado}"
        super().__init__(f"corrida {run_id} {situacion}: no admite {operacion}")


class EvidenciaInconsistenteError(PersistenciaEvidenciaError):
    """La evidencia no corresponde a la corrida o esta incompleta para publicar."""
