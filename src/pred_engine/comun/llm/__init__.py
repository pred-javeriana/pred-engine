"""Capa LLM desacoplada: protocolo, fabrica y tres proveedores HTTP."""

from pred_engine.comun.llm.catalogo import (
    AVAILABLE_MODELS,
    DEFAULT_MODELS,
    format_models_help,
    get_available_models,
)
from pred_engine.comun.llm.errores import (
    LlmError,
    LlmProviderError,
    LlmTimeoutError,
    UnknownModelError,
    UnknownProviderError,
)
from pred_engine.comun.llm.fabrica import (
    build_llm_provider,
    normalize_provider_name,
    resolve_model,
)
from pred_engine.comun.llm.protocolo import LlmProvider
from pred_engine.comun.llm.proveedores import (
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
)

__all__ = [
    "AVAILABLE_MODELS",
    "DEFAULT_MODELS",
    "AnthropicProvider",
    "GeminiProvider",
    "LlmError",
    "LlmProvider",
    "LlmProviderError",
    "LlmTimeoutError",
    "OpenAIProvider",
    "UnknownModelError",
    "UnknownProviderError",
    "build_llm_provider",
    "format_models_help",
    "get_available_models",
    "normalize_provider_name",
    "resolve_model",
]
