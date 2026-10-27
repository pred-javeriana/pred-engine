"""Handoff 1.4: contrato final, preservacion y publicacion Parquet."""

from pred_engine.ingesta.salida.errores import (
    HandoffPreconditionError,
    MissingCutoffMetadataError,
    OutputContractError,
    OutputHandoffError,
    PanelPreservationError,
)
from pred_engine.ingesta.salida.precondiciones import (
    evaluar_precondiciones_y_exclusiones,
    require_positive_demand,
)
from pred_engine.ingesta.salida.preservacion import require_panel_preserved
from pred_engine.ingesta.salida.publicador import (
    publish_classified_panel,
    read_classified_parquet,
    read_classified_parquet_with_sha256,
)
from pred_engine.ingesta.salida.validador import (
    normalize_output_frame,
    validate_output_contract,
)

__all__ = [
    "HandoffPreconditionError",
    "MissingCutoffMetadataError",
    "OutputContractError",
    "OutputHandoffError",
    "PanelPreservationError",
    "evaluar_precondiciones_y_exclusiones",
    "normalize_output_frame",
    "publish_classified_panel",
    "read_classified_parquet",
    "read_classified_parquet_with_sha256",
    "require_panel_preserved",
    "require_positive_demand",
    "validate_output_contract",
]
