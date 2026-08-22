"""Pruebas de la fabrica de proveedores LLM."""

from __future__ import annotations

import pytest

from pred_engine.comun.llm import (
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
    UnknownProviderError,
    build_llm_provider,
    normalize_provider_name,
)


def test_alias_claude_es_anthropic() -> None:
    assert normalize_provider_name("Claude") == "anthropic"


def test_proveedor_desconocido() -> None:
    with pytest.raises(UnknownProviderError, match="desconocido"):
        build_llm_provider("cohere", "sk-test")


def test_fabrica_devuelve_clases_correctas() -> None:
    assert isinstance(build_llm_provider("gemini", "k"), GeminiProvider)
    assert isinstance(build_llm_provider("openai", "k"), OpenAIProvider)
    assert isinstance(build_llm_provider("anthropic", "k"), AnthropicProvider)


def test_rechaza_api_key_vacia() -> None:
    with pytest.raises(ValueError, match="vacia"):
        build_llm_provider("gemini", "   ")
