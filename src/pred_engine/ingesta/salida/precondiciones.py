"""Gates fail-closed que deben ejecutarse antes de invocar al clasificador."""

from __future__ import annotations

import pandas as pd

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos import CANONICAL_FIELDS
from pred_engine.ingesta.salida.errores import HandoffPreconditionError

_logger = get_logger(__name__)


def require_positive_demand(panel: pd.DataFrame) -> None:
    """Exige al menos una demanda estrictamente positiva por SKU.

    No muta el panel. No calcula ADI ni CV²: solo cuenta d>0 para no
    iniciar la clasificacion sobre series matematicamente indefinidas.
    """
    if panel.empty:
        _logger.error("Handoff 1.4: panel diario vacio")
        raise HandoffPreconditionError("el panel diario no tiene filas")
    faltan = [c for c in ("sku_id", "demand_qty") if c not in panel.columns]
    if faltan:
        raise HandoffPreconditionError(f"faltan columnas {faltan}")
    extras_canonico = [c for c in CANONICAL_FIELDS if c not in panel.columns]
    if extras_canonico:
        raise HandoffPreconditionError(f"faltan columnas canonicas {extras_canonico}")

    positivos = panel.loc[panel["demand_qty"] > 0.0, "sku_id"].astype("string")
    todos = panel["sku_id"].astype("string")
    sin_demanda = sorted(set(todos.dropna()) - set(positivos.dropna()))
    if sin_demanda:
        # Solo identificadores de SKU, nunca magnitudes de demanda.
        _logger.error(
            "Handoff 1.4: %s SKU sin demanda estrictamente positiva",
            len(sin_demanda),
        )
        raise HandoffPreconditionError(
            "SKU sin demanda estrictamente positiva: " + ", ".join(sin_demanda)
        )
    _logger.info(
        "Precondicion de demanda positiva superada (skus=%s filas=%s)",
        int(len({str(v) for v in todos.tolist() if pd.notna(v)})),
        int(len(panel)),
    )
