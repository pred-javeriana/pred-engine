"""Gates fail-closed que deben ejecutarse antes de invocar al clasificador."""

from __future__ import annotations

import pandas as pd

from pred_engine.comun.exclusiones import SkuExclusion
from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos import CANONICAL_FIELDS
from pred_engine.comun.reserva import ReserveCut
from pred_engine.ingesta.salida.errores import HandoffPreconditionError

_logger = get_logger(__name__)


def evaluar_precondiciones_y_exclusiones(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, ReserveCut, tuple[SkuExclusion, ...]]:
    """Evalua precondiciones de ingesta y excluye SKU tardios con causa.

    Un producto es de lanzamiento tardio si no registra demanda estrictamente positiva
    en o antes de t*. Si todos los productos son tardios, rechaza el panel con
    HandoffPreconditionError.
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
    positivos = set(
        admisible.loc[admisible["demand_qty"] > 0.0, "sku_id"]
        .astype("string")
        .dropna()
        .unique()
    )
    todos = set(panel["sku_id"].astype("string").dropna().unique())
    sin_demanda = sorted(todos - positivos)

    if sin_demanda and len(sin_demanda) == len(todos):
        _logger.error(
            "Handoff 1.4: %s SKU sin demanda estrictamente positiva",
            len(sin_demanda),
        )
        raise HandoffPreconditionError(
            f"SKU sin demanda estrictamente positiva hasta t*={corte.t_star.date()}: "
            + ", ".join(sin_demanda)
        )

    exclusiones = tuple(
        SkuExclusion(sku_id=str(sku), cause="lanzado despues del corte")
        for sku in sin_demanda
    )
    admitido = panel.loc[~panel["sku_id"].astype("string").isin(sin_demanda)]
    _logger.info(
        "Precondiciones superadas: admitidos=%s excluidos=%s filas=%s t*=%s",
        len(todos) - len(sin_demanda),
        len(sin_demanda),
        len(admitido),
        corte.t_star.date(),
    )
    return admitido, corte, exclusiones


def require_positive_demand(panel: pd.DataFrame) -> None:
    """Exige al menos un producto con demanda positiva hasta t* en el panel."""
    evaluar_precondiciones_y_exclusiones(panel)
