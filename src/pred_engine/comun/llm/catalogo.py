"""Catalogo de modelos LLM permitidos por proveedor (tier economico, ago 2026)."""

from __future__ import annotations

# Modelos baratos curados por proveedor. El CLI y la fabrica solo aceptan IDs de aqui.
# Fuentes: ai.google.dev/pricing, developers.openai.com, docs.anthropic.com (ago 2026).
AVAILABLE_MODELS: dict[str, tuple[str, ...]] = {
    "gemini": (
        "gemini-2.5-flash-lite",  # mas barato estable (~$0.10/$0.40 por 1M tokens)
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-3-flash-preview",
        "gemini-3.5-flash",  # recomendado para pruebas reales del equipo
        "gemini-2.0-flash",
    ),
    "openai": (
        "gpt-4.1-nano",  # mas barato estable (~$0.10/$0.40 por 1M tokens)
        "gpt-5.4-nano",
        "gpt-5-mini",
        "gpt-4.1-mini",
        "gpt-4o-mini",
    ),
    "anthropic": (
        "claude-haiku-4-5",  # mas barato (~$1/$5 por 1M tokens)
        "claude-haiku-4-5-20251001",
    ),
}

# Default por proveedor: el modelo mas economico del catalogo (override con --model).
DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash-lite",
    "openai": "gpt-4.1-nano",
    "anthropic": "claude-haiku-4-5",
}


def get_available_models(provider: str) -> tuple[str, ...]:
    """Devuelve los modelos permitidos para un proveedor canonico."""
    return AVAILABLE_MODELS[provider]


def format_models_help(provider: str) -> str:
    """Texto para argparse/CLI: lista modelos y marca el default."""
    modelos = AVAILABLE_MODELS[provider]
    defecto = DEFAULT_MODELS[provider]
    lineas = [f"  {m}{' (default)' if m == defecto else ''}" for m in modelos]
    return "\n".join(lineas)
