"""Errores de la capa LLM compartida."""

from __future__ import annotations


class LlmError(RuntimeError):
    """Fallo de proveedor LLM (red, contrato HTTP o respuesta inutilizable)."""


class LlmTimeoutError(LlmError):
    """La API no respondio dentro del timeout configurado."""


class LlmProviderError(LlmError):
    """Error HTTP o payload inesperado del proveedor."""

    def __init__(self, mensaje: str, *, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(mensaje)


class UnknownProviderError(ValueError):
    """Nombre de proveedor no soportado por la fabrica."""


class UnknownModelError(ValueError):
    """Modelo no listado en el catalogo del proveedor."""
