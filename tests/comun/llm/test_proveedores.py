"""Pruebas HTTP de proveedores con httpx parcheado (sin red)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from pred_engine.comun.llm import (
    GeminiProvider,
    LlmProviderError,
    LlmTimeoutError,
    OpenAIProvider,
)
from pred_engine.comun.llm.proveedores.anthropic import AnthropicProvider


def _cliente_falso(cuerpo: dict[str, Any], status_code: int = 200) -> MagicMock:
    respuesta = MagicMock()
    respuesta.status_code = status_code
    respuesta.json.return_value = cuerpo
    if status_code >= 400:
        respuesta.raise_for_status.side_effect = httpx.HTTPStatusError(
            "err",
            request=MagicMock(),
            response=respuesta,
        )
    else:
        respuesta.raise_for_status.return_value = None
    cliente = MagicMock()
    cliente.post.return_value = respuesta
    cliente.__enter__.return_value = cliente
    cliente.__exit__.return_value = False
    return cliente


def test_gemini_extrae_texto_y_usa_temperatura_cero() -> None:
    cuerpo = {
        "candidates": [{"content": {"parts": [{"text": '{"sku_id": "Item_ID"}'}]}}]
    }
    cliente = _cliente_falso(cuerpo)
    with patch("pred_engine.comun.llm.http.httpx.Client", return_value=cliente):
        texto = GeminiProvider("secret-key", model="gemini-2.0-flash").complete(
            "prompt",
            temperature=0.0,
            timeout=5.0,
        )
    assert "Item_ID" in texto
    kwargs = cliente.post.call_args.kwargs
    assert kwargs["json"]["generationConfig"]["temperature"] == 0.0
    assert kwargs["params"]["key"] == "secret-key"


def test_timeout_se_traduce_a_llm_timeout_error() -> None:
    cliente = MagicMock()
    cliente.__enter__.return_value = cliente
    cliente.__exit__.return_value = False
    cliente.post.side_effect = httpx.TimeoutException("t")
    with patch("pred_engine.comun.llm.http.httpx.Client", return_value=cliente):
        with pytest.raises(LlmTimeoutError, match="no respondio"):
            OpenAIProvider("sk", model="gpt-4o-mini").complete(
                "p",
                temperature=0.0,
                timeout=1.0,
            )


def test_http_error_no_incluye_la_clave() -> None:
    cliente = _cliente_falso({}, status_code=401)
    with patch("pred_engine.comun.llm.http.httpx.Client", return_value=cliente):
        with pytest.raises(LlmProviderError, match="401") as captured:
            AnthropicProvider("super-secret-key", model="claude-sonnet-4-5").complete(
                "p",
                temperature=0.0,
                timeout=5.0,
            )
    assert "super-secret-key" not in str(captured.value)
