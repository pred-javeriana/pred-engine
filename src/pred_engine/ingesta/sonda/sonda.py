"""Orquestacion de la sonda: muestra → LLM temp 0.0 → diagnostico JSON (sin mutar)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from pydantic import ValidationError

from pred_engine.comun.llm import LlmProvider, LlmTimeoutError
from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos import HeaderDiagnostic
from pred_engine.ingesta.sonda.errores import SemanticAlignmentError
from pred_engine.ingesta.sonda.mapeo import DiagnosticArtifact
from pred_engine.ingesta.sonda.prompt import N_SAMPLE_ROWS, build_alignment_prompt

_logger = get_logger(__name__)
_TEMPERATURA = 0.0
_CERCA = "`" * 3


def parse_header_diagnostic(texto: str) -> HeaderDiagnostic:
    """Parsea JSON del LLM. Rechaza basura y payloads fuera del contrato."""
    crudo = texto.strip()
    if crudo.startswith(_CERCA):
        crudo = crudo.strip("`")
        if crudo.startswith("json"):
            crudo = crudo[4:]
        crudo = crudo.strip()
    try:
        carga: Any = json.loads(crudo)
    except json.JSONDecodeError as exc:
        _logger.error("La sonda LLM no devolvio JSON valido")
        raise SemanticAlignmentError(
            "PRED no puede trabajar con este dataset: la sonda no devolvio "
            "un JSON de diagnostico valido."
        ) from exc
    if not isinstance(carga, dict):
        raise SemanticAlignmentError(
            "PRED no puede trabajar con este dataset: el diagnostico no es un "
            "objeto JSON."
        )
    try:
        return HeaderDiagnostic.model_validate(carga)
    except ValidationError as exc:
        _logger.error("JSON de diagnostico no coincide con HeaderDiagnostic")
        raise SemanticAlignmentError(
            "PRED no puede trabajar con este dataset: el JSON de diagnostico "
            "no coincide con el contrato."
        ) from exc


def probe_headers(
    frame: pd.DataFrame,
    provider: LlmProvider,
    *,
    timeout: float = 30.0,
    n_rows: int = N_SAMPLE_ROWS,
) -> DiagnosticArtifact:
    """Sonda consultiva sin estado. No muta el marco. Fail-fast si rejected."""
    columnas_originales = list(frame.columns)
    prompt = build_alignment_prompt(frame, n_rows=n_rows)
    try:
        texto = provider.complete(
            prompt,
            temperature=_TEMPERATURA,
            timeout=timeout,
        )
    except LlmTimeoutError:
        _logger.exception("Timeout de la sonda de cabeceras tras %.1fs", timeout)
        raise
    diagnostico = parse_header_diagnostic(texto)
    if diagnostico.is_rejected():
        _logger.error(
            "Sonda rechazada: %s",
            diagnostico.model_dump(),
        )
        raise SemanticAlignmentError(
            "PRED no puede ingerir este dataset: la sonda rechazo las cabeceras. "
            "Corrija el CSV segun el reporte JSON de diagnostico.",
            diagnostic=diagnostico,
        )
    if list(frame.columns) != columnas_originales:
        raise SemanticAlignmentError(
            "PRED no puede ingerir este dataset: la sonda muto el marco "
            "(violacion de politica consultiva)."
        )
    _logger.info(
        "Sonda aceptada (%s entradas de diagnostico)",
        len(diagnostico.diagnostic),
    )
    return DiagnosticArtifact(frame=frame, diagnostic=diagnostico)
