"""Capa LLM desacoplada: protocolo, fabrica y tres proveedores HTTP."""

from pred_engine.comun.llm.errores import (
    LlmError,
    LlmProviderError,
    LlmTimeoutError,
    UnknownProviderError,
)
from pred_engine.comun.llm.fabrica import (
    DEFAULT_MODELS,
    build_llm_provider,
    normalize_provider_name,
)
from pred_engine.comun.llm.protocolo import LlmProvider
from pred_engine.comun.llm.proveedores import (
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
)

__all__ = [
    "DEFAULT_MODELS",
    "AnthropicProvider",
    "GeminiProvider",
    "LlmError",
    "LlmProvider",
    "LlmProviderError",
    "LlmTimeoutError",
    "OpenAIProvider",
    "UnknownProviderError",
    "build_llm_provider",
    "normalize_provider_name",
]
