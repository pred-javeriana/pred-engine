"""Inyeccion de sku_class a nivel de SKU sobre el panel diario."""

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
from pred_engine.ingesta.categorizacion.adi import compute_adi
from pred_engine.ingesta.categorizacion.cv2 import compute_cv2
from pred_engine.ingesta.categorizacion.enrutador import route_syntetos_boylan
from pred_engine.ingesta.categorizacion.errores import TopologyContractError

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TopologyArtifact:
    """Panel enriquecido mas metricas por SKU (no se serializan al Parquet)."""

    frame: pd.DataFrame
    metrics: tuple[TopologyMetrics, ...]


def classify_panel(frame: pd.DataFrame) -> TopologyArtifact:
    """Agrupa por sku_id, clasifica una vez y rellena sku_class por broadcast."""
    if frame.empty:
        raise TopologyContractError("no hay filas para clasificar")
    faltan = [c for c in CANONICAL_FIELDS if c not in frame.columns]
    if faltan:
        raise TopologyContractError(f"faltan columnas {faltan}")

    metricas: list[TopologyMetrics] = []
    clases: dict[str, str] = {}

    for sku, grupo in frame.groupby("sku_id", sort=True):
        demanda = grupo["demand_qty"].to_numpy(dtype="float64", copy=True)
        adi = compute_adi(demanda)
        cv2 = compute_cv2(demanda)
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
