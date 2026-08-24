"""Adaptador Anthropic (messages API)."""

from __future__ import annotations

from typing import Any

from pred_engine.comun.llm.errores import LlmProviderError
from pred_engine.comun.llm.http import post_json

_URL = "https://api.anthropic.com/v1/messages"


class AnthropicProvider:
    """Cliente sin estado contra Messages."""

    def __init__(self, api_key: str, *, model: str) -> None:
        if not api_key.strip():
            raise ValueError("api_key de Anthropic vacia")
        self._api_key = api_key
        self._model = model

    def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        timeout: float,
    ) -> str:
        payload = {
            "model": self._model,
            "max_tokens": 1024,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        cuerpo = post_json(
            _URL,
            payload=payload,
            timeout=timeout,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )
        return _texto_anthropic(cuerpo)


def _texto_anthropic(cuerpo: dict[str, Any]) -> str:
    bloques = cuerpo.get("content") or []
    textos = [
        b.get("text", "")
        for b in bloques
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    texto = "".join(textos).strip()
    if not texto:
        raise LlmProviderError("Anthropic devolvio texto vacio")
    return texto
