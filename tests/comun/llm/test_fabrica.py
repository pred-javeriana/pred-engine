"""Pruebas de la fabrica de proveedores LLM."""

from __future__ import annotations

import pytest

from pred_engine.comun.llm import (
    AVAILABLE_MODELS,
    DEFAULT_MODELS,
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
    UnknownModelError,
    UnknownProviderError,
    build_llm_provider,
    normalize_provider_name,
    resolve_model,
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


def test_defaults_son_tier_economico() -> None:
    assert DEFAULT_MODELS["gemini"] == "gemini-2.5-flash-lite"
    assert DEFAULT_MODELS["openai"] == "gpt-4.1-nano"
    assert DEFAULT_MODELS["anthropic"] == "claude-haiku-4-5"


def test_gemini_3_5_flash_esta_en_catalogo() -> None:
    assert "gemini-3.5-flash" in AVAILABLE_MODELS["gemini"]


def test_acepta_gemini_3_5_flash_explicito() -> None:
    proveedor = build_llm_provider(
        "gemini",
        "k",
        model="gemini-3.5-flash",
    )
    assert isinstance(proveedor, GeminiProvider)
    assert proveedor._model == "gemini-3.5-flash"


def test_rechaza_modelo_fuera_de_catalogo() -> None:
    with pytest.raises(UnknownModelError, match="no permitido"):
        build_llm_provider("openai", "k", model="gpt-99-ultra")


def test_resolve_model_usa_default_si_none() -> None:
    assert resolve_model("gemini", None) == "gemini-2.5-flash-lite"


def test_catalogo_tiene_opciones_por_proveedor() -> None:
    assert len(AVAILABLE_MODELS["gemini"]) >= 5
    assert len(AVAILABLE_MODELS["openai"]) >= 4
    assert len(AVAILABLE_MODELS["anthropic"]) >= 2
