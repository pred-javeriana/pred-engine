"""Inyeccion de sku_class a nivel de SKU sobre el panel diario.

La clase se calcula solo con la historia anterior al corte t* de la reserva
(ADR-03-003, ADR-019): la demanda de los dias reservados para M3 no influye en
la clase ni, por lo tanto, en la ruta de familias de M2. El panel publicado
conserva todas las filas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos import (
    CANONICAL_FIELDS,
    PANEL_FIELDS,
    TopologyMetrics,
)
from pred_engine.comun.reserva import ReserveCut
from pred_engine.ingesta.categorizacion.adi import compute_adi
from pred_engine.ingesta.categorizacion.cv2 import compute_cv2
from pred_engine.ingesta.categorizacion.enrutador import route_syntetos_boylan
from pred_engine.ingesta.categorizacion.errores import (
    TopologyContractError,
    TopologyMathError,
)

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TopologyArtifact:
    """Panel enriquecido mas metricas por SKU (no se serializan al Parquet)."""

    frame: pd.DataFrame
    metrics: tuple[TopologyMetrics, ...]


def classify_panel(frame: pd.DataFrame) -> TopologyArtifact:
    """Clasifica cada SKU con su historia hasta t* y rellena sku_class por broadcast."""
    if frame.empty:
        raise TopologyContractError("no hay filas para clasificar")
    faltan = [c for c in CANONICAL_FIELDS if c not in frame.columns]
    if faltan:
        raise TopologyContractError(f"faltan columnas {faltan}")
    try:
        corte = ReserveCut.of(frame)
    except ValueError as exc:
        raise TopologyContractError(str(exc)) from exc
    dias = pd.to_datetime(frame["timestamp"]).dt.normalize()
    admisible = frame.loc[dias <= corte.t_star]
    _logger.info(
        "Clasificacion sobre la historia admisible t*=%s dias_reservados=%s",
        corte.t_star.date(),
        corte.reserved_days,
    )

    metricas: list[TopologyMetrics] = []
    clases: dict[str, str] = {}

    for sku in sorted(frame["sku_id"].astype("string").unique()):
        grupo = admisible.loc[admisible["sku_id"].astype("string") == sku]
        demanda = grupo["demand_qty"].to_numpy(dtype="float64", copy=True)
        try:
            adi = compute_adi(demanda)
            cv2 = compute_cv2(demanda)
        except TopologyMathError as exc:
            raise TopologyMathError(
                f"SKU {sku}: sin demanda positiva hasta t*={corte.t_star.date()}; "
                f"no se clasifica con los dias reservados ({exc})"
            ) from exc
        n_periodos = int(np.isfinite(demanda).sum())
        n_positivos = int(np.sum(demanda > 0.0))
        clase = route_syntetos_boylan(adi, cv2)
        sku_texto = str(sku)
        metricas.append(
            TopologyMetrics(
                sku_id=sku_texto,
                n_periods=n_periodos,
                n_positive=n_positivos,
                adi=adi,
                cv2=cv2,
                sku_class=clase,
            )
        )
        clases[sku_texto] = clase
        _logger.info(
            "SKU clasificado id=%s clase=%s n_periodos=%s n_pos=%s",
            sku_texto,
            clase,
            n_periodos,
            n_positivos,
        )

    panel = frame.loc[:, list(CANONICAL_FIELDS)].copy()
    panel["sku_class"] = panel["sku_id"].astype("string").map(clases)
    panel["sku_class"] = panel["sku_class"].astype("string")
    if bool(panel["sku_class"].isna().any()):
        raise TopologyContractError("hubo SKU sin etiqueta sku_class")

    conteos = panel.groupby("sku_id", sort=True)["sku_class"].nunique()
    if not bool((conteos == 1).all()):
        raise TopologyContractError("un SKU recibio mas de una etiqueta sku_class")

    return TopologyArtifact(
        frame=panel.loc[:, list(PANEL_FIELDS)],
        metrics=tuple(metricas),
    )


def classify_daily_panel(frame: pd.DataFrame) -> TopologyArtifact:
    """Alias del contrato Notion 1.4: delega al clasificador real de 1.3."""
    return classify_panel(frame)
