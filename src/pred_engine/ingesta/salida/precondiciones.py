"""Gates fail-closed que deben ejecutarse antes de invocar al clasificador."""

from __future__ import annotations

from typing import cast

import pandas as pd

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos import CANONICAL_FIELDS
from pred_engine.comun.reserva import ReserveCut
from pred_engine.ingesta.salida.errores import HandoffPreconditionError
from pred_engine.ingesta.salida.validador import contar_skus

_logger = get_logger(__name__)


def require_positive_demand(panel: pd.DataFrame) -> None:
    """Exige al menos una demanda estrictamente positiva por SKU hasta t*.

    No muta el panel. No calcula ADI ni CV²: solo cuenta d>0 en la historia
    que usa el clasificador (sin los dias reservados, ADR-019) para no iniciar
    la clasificacion sobre series matematicamente indefinidas.
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

    try:
        corte = ReserveCut.of(panel)
    except ValueError as exc:
        raise HandoffPreconditionError(str(exc)) from exc
    dias = pd.to_datetime(panel["timestamp"]).dt.normalize()
    admisible = panel.loc[dias <= corte.t_star]
    positivos = admisible.loc[admisible["demand_qty"] > 0.0, "sku_id"].astype("string")
    todos = panel["sku_id"].astype("string")
    sin_demanda = sorted(
        set(todos.dropna().unique()) - set(positivos.dropna().unique())
    )
    if sin_demanda:
        # Solo identificadores de SKU, nunca magnitudes de demanda.
        _logger.error(
            "Handoff 1.4: %s SKU sin demanda estrictamente positiva",
            len(sin_demanda),
        )
        raise HandoffPreconditionError(
            f"SKU sin demanda estrictamente positiva hasta t*={corte.t_star.date()}: "
            + ", ".join(sin_demanda)
        )
    _logger.info(
        "Precondicion de demanda positiva superada (skus=%s filas=%s)",
        contar_skus(cast(pd.Series, todos)),
        int(len(panel)),
    )
