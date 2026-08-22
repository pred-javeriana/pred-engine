"""Orquestacion de la sonda: muestra → LLM temp 0.0 → parseo → rename."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from pydantic import ValidationError

from pred_engine.comun.llm import LlmProvider, LlmTimeoutError
from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos import HeaderMapping
from pred_engine.ingesta.sonda.errores import SemanticAlignmentError
from pred_engine.ingesta.sonda.mapeo import AlignmentArtifact, apply_header_mapping
from pred_engine.ingesta.sonda.prompt import N_SAMPLE_ROWS, build_alignment_prompt

_logger = get_logger(__name__)
_TEMPERATURA = 0.0
_CERCA = "`" * 3


def parse_header_mapping(texto: str) -> HeaderMapping:
    """Parsea JSON del LLM. Rechaza basura y campos inventados como mapeo."""
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
            "un JSON de mapeo valido."
        ) from exc
    if not isinstance(carga, dict):
        raise SemanticAlignmentError(
            "PRED no puede trabajar con este dataset: el mapeo no es un objeto JSON."
        )
    try:
        return HeaderMapping.model_validate(carga)
    except ValidationError as exc:
        _logger.error("JSON de mapeo no coincide con HeaderMapping")
        raise SemanticAlignmentError(
            "PRED no puede trabajar con este dataset: el JSON de mapeo "
            "no coincide con el contrato."
        ) from exc


def probe_headers(
    frame: pd.DataFrame,
    provider: LlmProvider,
    *,
    timeout: float = 30.0,
    n_rows: int = N_SAMPLE_ROWS,
) -> AlignmentArtifact:
    """Sonda sin estado. Propaga LlmTimeoutError tras registrarlo."""
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
    mapeo = parse_header_mapping(texto)
    return apply_header_mapping(frame, mapeo)
