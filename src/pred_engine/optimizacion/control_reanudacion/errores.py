"""Errores tipados del control de reanudacion (fail-closed, sin PII)."""

from __future__ import annotations


class ReanudacionError(Exception):
    """Frontera 2.9: la corrida no puede crearse, reanudarse o persistirse."""


class TransicionEstadoError(ReanudacionError, ValueError):
    """La maquina de estados rechazo el cambio pedido."""


class IncompatibilidadCorridaError(ReanudacionError):
    """La solicitud no coincide con el manifiesto persistido."""

    def __init__(self, motivos: tuple[str, ...]) -> None:
        self.motivos = motivos
        detalle = "; ".join(motivos) if motivos else "huella distinta"
        super().__init__(f"corrida incompatible: {detalle}")


class CorridaFallidaError(ReanudacionError):
    """Se intento reabrir una corrida marcada como fallida."""


class ManifiestoAusenteError(ReanudacionError):
    """No hay manifiesto en la ruta esperada."""


class ManifiestoCorruptoError(ReanudacionError):
    """El artefacto existe pero no es un manifiesto valido."""


class VersionManifiestoError(ReanudacionError):
    """`schema_version` no coincide con la version soportada."""
