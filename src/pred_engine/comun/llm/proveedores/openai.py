"""Adaptador OpenAI (chat.completions + json_object)."""

from __future__ import annotations

from typing import Any

from pred_engine.comun.llm.errores import LlmProviderError
from pred_engine.comun.llm.http import post_json

_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIProvider:
    """Cliente sin estado contra Chat Completions."""

    def __init__(self, api_key: str, *, model: str) -> None:
        if not api_key.strip():
            raise ValueError("api_key de OpenAI vacia")
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
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        }
        cuerpo = post_json(
            _URL,
            payload=payload,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        return _texto_openai(cuerpo)


def _texto_openai(cuerpo: dict[str, Any]) -> str:
    choices = cuerpo.get("choices") or []
    if not choices:
        raise LlmProviderError("OpenAI no devolvio choices")
    texto = ((choices[0].get("message") or {}).get("content") or "").strip()
    if not texto:
        raise LlmProviderError("OpenAI devolvio texto vacio")
    return texto
