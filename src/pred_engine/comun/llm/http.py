"""POST JSON compartido: timeout unico, sin loguear URLs con API keys."""

from __future__ import annotations

from typing import Any

import httpx

from pred_engine.comun.llm.errores import LlmProviderError, LlmTimeoutError
from pred_engine.comun.logger import get_logger

_logger = get_logger(__name__)


def _detalle_http(respuesta: httpx.Response) -> str:
    """Extrae un mensaje corto del cuerpo de error sin filtrar secretos."""
    try:
        cuerpo = respuesta.json()
    except ValueError:
        texto = respuesta.text.strip()
        return texto[:240] if texto else respuesta.reason_phrase
    if isinstance(cuerpo, dict):
        error = cuerpo.get("error")
        if isinstance(error, dict):
            mensaje = error.get("message")
            if isinstance(mensaje, str) and mensaje.strip():
                return mensaje.strip()
        mensaje = cuerpo.get("message")
        if isinstance(mensaje, str) and mensaje.strip():
            return mensaje.strip()
    return respuesta.reason_phrase


def post_json(
    url: str,
    *,
    payload: dict[str, Any],
    timeout: float,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Ejecuta un POST y devuelve el objeto JSON. No registra query params."""
    try:
        with httpx.Client(timeout=timeout) as cliente:
            respuesta = cliente.post(
                url,
                json=payload,
                headers=headers,
                params=params,
            )
            respuesta.raise_for_status()
            cuerpo = respuesta.json()
    except httpx.TimeoutException as exc:
        _logger.error("Timeout LLM tras %.1fs (url host-only)", timeout)
        raise LlmTimeoutError(
            f"El proveedor LLM no respondio en {timeout:.1f}s"
        ) from exc
    except httpx.HTTPStatusError as exc:
        codigo = exc.response.status_code
        detalle = _detalle_http(exc.response)
        _logger.error("Proveedor LLM HTTP %s: %s", codigo, detalle)
        raise LlmProviderError(
            f"El proveedor LLM respondio HTTP {codigo}: {detalle}",
            status_code=codigo,
        ) from exc
    except httpx.HTTPError as exc:
        _logger.error("Fallo de red hacia el proveedor LLM")
        raise LlmProviderError("Fallo de red hacia el proveedor LLM") from exc
    except ValueError as exc:
        _logger.error("El proveedor LLM no devolvio JSON valido")
        raise LlmProviderError("El proveedor LLM no devolvio un objeto JSON") from exc
    if not isinstance(cuerpo, dict):
        raise LlmProviderError("El proveedor LLM no devolvio un objeto JSON")
    return cuerpo
