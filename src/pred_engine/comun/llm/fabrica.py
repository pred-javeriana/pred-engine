"""Fabrica de proveedores a partir de un nombre estable (gemini|openai|anthropic)."""

from __future__ import annotations

from pred_engine.comun.llm.catalogo import (
    AVAILABLE_MODELS,
    DEFAULT_MODELS,
    format_models_help,
)
from pred_engine.comun.llm.errores import UnknownModelError, UnknownProviderError
from pred_engine.comun.llm.protocolo import LlmProvider
from pred_engine.comun.llm.proveedores import (
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
)

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


def resolve_model(provider: str, model: str | None) -> str:
    """Elige el modelo: default economico o uno explicito del catalogo."""
    canonico = normalize_provider_name(provider)
    elegido = (model or DEFAULT_MODELS[canonico]).strip()
    permitidos = AVAILABLE_MODELS[canonico]
    if elegido not in permitidos:
        raise UnknownModelError(
            f"Modelo LLM no permitido para {canonico!r}: {elegido!r}. "
            f"Opciones:\n{format_models_help(canonico)}"
        )
    return elegido


def build_llm_provider(
    name: str,
    api_key: str,
    *,
    model: str | None = None,
) -> LlmProvider:
    """Construye un adaptador. No abre conexiones hasta complete()."""
    canonico = normalize_provider_name(name)
    elegido = resolve_model(canonico, model)
    if canonico == "gemini":
        return GeminiProvider(api_key, model=elegido)
    if canonico == "openai":
        return OpenAIProvider(api_key, model=elegido)
    return AnthropicProvider(api_key, model=elegido)
