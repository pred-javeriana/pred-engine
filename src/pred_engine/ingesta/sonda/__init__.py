"""Sonda de cabeceras LLM y adaptador de mapeo vectorizado."""

from pred_engine.ingesta.sonda.errores import SemanticAlignmentError
from pred_engine.ingesta.sonda.mapeo import AlignmentArtifact, apply_header_mapping
from pred_engine.ingesta.sonda.prompt import (
    N_SAMPLE_ROWS,
    build_alignment_prompt,
    sample_header_frame,
)
from pred_engine.ingesta.sonda.sonda import parse_header_mapping, probe_headers

__all__ = [
    "N_SAMPLE_ROWS",
    "AlignmentArtifact",
    "SemanticAlignmentError",
    "apply_header_mapping",
    "build_alignment_prompt",
    "parse_header_mapping",
    "probe_headers",
    "sample_header_frame",
]
