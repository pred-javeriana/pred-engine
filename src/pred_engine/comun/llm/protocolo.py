"""Contrato de proveedor LLM (inyeccion de dependencias, sin estado)."""

from __future__ import annotations

from typing import Protocol


class LlmProvider(Protocol):
    """Adaptador HTTP de un proveedor. Implementaciones no retienen conversacion."""

    def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        timeout: float,
    ) -> str:
        """Devuelve el texto de la respuesta. Lanza LlmTimeoutError si expira."""
        ...
