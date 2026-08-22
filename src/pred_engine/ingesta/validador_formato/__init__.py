"""Validacion estricta de esquemas post-alineacion."""

from pred_engine.ingesta.validador_formato.barrera import validate_aligned_frame
from pred_engine.ingesta.validador_formato.errores import SchemaBarrierError

__all__ = ["SchemaBarrierError", "validate_aligned_frame"]
