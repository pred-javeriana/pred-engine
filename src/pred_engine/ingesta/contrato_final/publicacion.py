"""Contrato 1.4: panel clasificado por 1.3 -> artefacto publicado."""

from __future__ import annotations

from collections.abc import Callable
from typing import NoReturn

import pandas as pd

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos import (
    CANONICAL_FIELDS,
    HANDOFF_FIELDS,
    SKU_CLASS_LABELS,
)
from pred_engine.ingesta.contrato_final.errores import HandoffContractError

_logger = get_logger(__name__)

ClassifyDailyPanel = Callable[[pd.DataFrame], pd.DataFrame]


def require_positive_demand(panel: pd.DataFrame) -> pd.DataFrame:
    """Fail-closed sobre el panel diario. No clasifica ni publica SKU en ceros."""
    if panel.empty:
        _fallar("el panel diario no tiene filas")
    if "sku_id" not in panel.columns or "demand_qty" not in panel.columns:
        _fallar("el panel diario debe incluir sku_id y demand_qty")
    maximos = panel.groupby("sku_id", sort=False)["demand_qty"].max()
    vacios = [str(sku) for sku, maximo in maximos.items() if not (maximo > 0)]
    if vacios:
        _fallar(
            "calidad de datos: SKU sin periodos con demanda > 0 "
            "(ADI y CV2 indefinidos; el SKU no se publica): "
            + ", ".join(sorted(vacios))
        )
    return panel


def default_classify_daily_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Delega en el clasificador 1.3. No calcula ADI ni CV2."""
    try:
        from pred_engine.ingesta.categorizacion import classify_daily_panel
    except ImportError as exc:
        raise HandoffContractError(
            "classify_daily_panel no esta exportado por "
            "pred_engine.ingesta.categorizacion"
        ) from exc
    return classify_daily_panel(panel)


def enforce_handoff_contract(
    frame: pd.DataFrame,
    source_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Copia tipada del panel 1.4. No muta el original ni clasifica."""
    if not isinstance(frame, pd.DataFrame):
        _fallar("el panel clasificado debe ser un DataFrame")
    if list(frame.columns) != list(HANDOFF_FIELDS):
        _fallar(
            "el panel clasificado debe tener exactamente " + ", ".join(HANDOFF_FIELDS)
        )
    if frame.empty:
        _fallar("el panel clasificado no tiene filas")
    if source_panel is not None:
        _require_same_panel(source_panel, frame)
    if frame["sku_class"].isna().any():
        _fallar("sku_class no puede ser nulo")
    if bool((frame["demand_qty"] < 0).any()):
        _fallar("demand_qty negativa")
    if bool((frame["lead_time_days"] < 1).any()):
        _fallar("lead_time_days < 1")

    etiquetas = {str(valor) for valor in frame["sku_class"].tolist()}
    invalidas = etiquetas - set(SKU_CLASS_LABELS)
    if invalidas:
        _fallar(
            "sku_class contiene etiquetas no permitidas: "
            + ", ".join(sorted(invalidas))
        )

    por_sku = frame.groupby("sku_id", sort=False)["sku_class"].nunique()
    if bool((por_sku != 1).any()):
        _fallar("sku_class debe ser constante por sku_id")

    salida = frame.loc[:, list(HANDOFF_FIELDS)].copy()
    salida["sku_id"] = salida["sku_id"].astype("string")
    salida["timestamp"] = pd.to_datetime(salida["timestamp"]).astype("datetime64[ns]")
    salida["demand_qty"] = salida["demand_qty"].astype("float64")
    salida["lead_time_days"] = salida["lead_time_days"].astype("int64")
    salida["sku_class"] = salida["sku_class"].astype("string")
    _logger.info("Contrato 1.4 superado (%s filas)", len(salida))
    return salida


def _require_same_panel(source_panel: pd.DataFrame, classified: pd.DataFrame) -> None:
    if not isinstance(source_panel, pd.DataFrame):
        _fallar("el panel diario debe ser un DataFrame")
    if list(source_panel.columns) != list(CANONICAL_FIELDS):
        _fallar("el panel diario debe tener exactamente " + ", ".join(CANONICAL_FIELDS))
    expected = source_panel.loc[:, list(CANONICAL_FIELDS)].reset_index(drop=True)
    received = classified.loc[:, list(CANONICAL_FIELDS)].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            expected,
            received,
            check_dtype=False,
            check_names=False,
        )
    except AssertionError:
        _fallar("la clasificacion debe devolver las mismas filas del panel diario")


def _fallar(mensaje: str) -> NoReturn:
    _logger.error("Contrato de frontera: %s", mensaje)
    raise HandoffContractError(mensaje)
