"""Proyeccion pura al contrato canonico (extras ERP se descartan por nombre)."""

from __future__ import annotations

import pandas as pd

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos import CANONICAL_FIELDS
from pred_engine.ingesta.categorizacion.errores import TopologyContractError

_logger = get_logger(__name__)


def select_canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Copia con exactamente CANONICAL_FIELDS. No renombra. No muta origen."""
    presentes = [str(c) for c in frame.columns]
    faltan = [c for c in CANONICAL_FIELDS if c not in presentes]
    if faltan:
        raise TopologyContractError(f"faltan columnas canonicas {faltan}")
    extras = [c for c in presentes if c not in CANONICAL_FIELDS]
    if extras:
        # Solo nombres: no se registran valores de stock, costo ni proveedor.
        _logger.info("Columnas extra omitidas: %s", extras)
    return frame.loc[:, list(CANONICAL_FIELDS)].copy()
