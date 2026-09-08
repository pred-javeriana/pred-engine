"""Handoff 1.4: contrato final, preservacion y publicacion Parquet."""

from pred_engine.ingesta.salida.errores import (
    HandoffPreconditionError,
    OutputContractError,
    OutputHandoffError,
    PanelPreservationError,
)
from pred_engine.ingesta.salida.precondiciones import require_positive_demand
from pred_engine.ingesta.salida.preservacion import require_panel_preserved
from pred_engine.ingesta.salida.publicador import (
    publish_classified_panel,
    read_classified_parquet,
)
from pred_engine.ingesta.salida.validador import (
    normalize_output_frame,
    validate_output_contract,
)

__all__ = [
    "HandoffPreconditionError",
    "OutputContractError",
    "OutputHandoffError",
    "PanelPreservationError",
    "normalize_output_frame",
    "publish_classified_panel",
    "read_classified_parquet",
    "require_panel_preserved",
    "require_positive_demand",
    "validate_output_contract",
]
