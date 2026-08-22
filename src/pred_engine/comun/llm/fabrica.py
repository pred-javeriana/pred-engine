"""Fabrica de proveedores a partir de un nombre estable (gemini|openai|anthropic)."""

from __future__ import annotations

from pred_engine.comun.llm.errores import UnknownProviderError
from pred_engine.comun.llm.protocolo import LlmProvider
from pred_engine.comun.llm.proveedores import (
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
)

DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5",
}

_ALIAS: dict[str, str] = {
    "gemini": "gemini",
    "google": "gemini",
    "openai": "openai",
    "gpt": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
}


def normalize_provider_name(name: str) -> str:
    clave = name.strip().lower()
    if clave not in _ALIAS:
        raise UnknownProviderError(
            f"Proveedor LLM desconocido: {name!r}. Use gemini, openai o anthropic."
        )
    return _ALIAS[clave]


def build_llm_provider(
    name: str,
    api_key: str,
    *,
    model: str | None = None,
) -> LlmProvider:
    """Construye un adaptador. No abre conexiones hasta complete()."""
    canonico = normalize_provider_name(name)
    elegido = model or DEFAULT_MODELS[canonico]
    if canonico == "gemini":
        return GeminiProvider(api_key, model=elegido)
    if canonico == "openai":
        return OpenAIProvider(api_key, model=elegido)
    return AnthropicProvider(api_key, model=elegido)
