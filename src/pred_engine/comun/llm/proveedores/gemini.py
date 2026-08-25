"""Adaptador Gemini (Google AI Studio, generateContent)."""

from __future__ import annotations

from typing import Any

from pred_engine.comun.llm.errores import LlmProviderError
from pred_engine.comun.llm.http import post_json

_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Schema OpenAPI minimo: diagnostico consultivo (sin mapeo automatico).
_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["accepted", "rejected"]},
        "diagnostic": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "severity": {"type": "string", "enum": ["error", "info"]},
                    "message": {"type": "string"},
                    "action": {"type": "string", "nullable": True},
                },
                "required": ["field", "message"],
            },
        },
    },
    "required": ["status", "diagnostic"],
}


class GeminiProvider:
    """Cliente sin estado contra generateContent."""

    def __init__(self, api_key: str, *, model: str) -> None:
        if not api_key.strip():
            raise ValueError("api_key de Gemini vacia")
        self._api_key = api_key
        self._model = model

    def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        timeout: float,
    ) -> str:
        url = _URL.format(model=self._model)
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "candidateCount": 1,
                "responseMimeType": "application/json",
                "responseSchema": _RESPONSE_SCHEMA,
            },
        }
        cuerpo = post_json(
            url,
            payload=payload,
            timeout=timeout,
            params={"key": self._api_key},
        )
        return _texto_gemini(cuerpo)


def _texto_gemini(cuerpo: dict[str, Any]) -> str:
    candidatos = cuerpo.get("candidates") or []
    if not candidatos:
        raise LlmProviderError("Gemini no devolvio candidatos")
    partes = (candidatos[0].get("content") or {}).get("parts") or []
    textos = [p.get("text", "") for p in partes if isinstance(p, dict)]
    texto = "".join(textos).strip()
    if not texto:
        raise LlmProviderError("Gemini devolvio texto vacio")
    return texto
