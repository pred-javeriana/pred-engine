"""Sonda de cabeceras LLM consultiva (diagnostico JSON, sin mutacion)."""

from pred_engine.ingesta.sonda.errores import SemanticAlignmentError
from pred_engine.ingesta.sonda.mapeo import DiagnosticArtifact
from pred_engine.ingesta.sonda.prompt import (
    N_SAMPLE_ROWS,
    build_alignment_prompt,
    sample_header_frame,
)
from pred_engine.ingesta.sonda.sonda import parse_header_diagnostic, probe_headers

__all__ = [
    "N_SAMPLE_ROWS",
    "DiagnosticArtifact",
    "SemanticAlignmentError",
    "build_alignment_prompt",
    "parse_header_diagnostic",
    "probe_headers",
    "sample_header_frame",
]
