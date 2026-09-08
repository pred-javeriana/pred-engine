"""Comprueba que 1.3 solo anadio sku_class al panel diario."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos import CANONICAL_FIELDS, PANEL_FIELDS, TOPOLOGY_FIELD
from pred_engine.ingesta.salida.errores import PanelPreservationError

_logger = get_logger(__name__)


def _fallar(mensaje: str) -> None:
    _logger.error("Preservacion 1.4 violada: %s", mensaje)
    raise PanelPreservationError(mensaje)


def _como_texto(serie: pd.Series) -> np.ndarray:
    return np.asarray(serie.astype("string").to_numpy(copy=False))


def _como_float(serie: pd.Series) -> np.ndarray:
    return np.asarray(serie.to_numpy(copy=True), dtype=np.float64)


def _como_timestamp_ns(serie: pd.Series) -> np.ndarray:
    convertidos = pd.to_datetime(serie, utc=False)
    return np.asarray(convertidos.to_numpy(), dtype="datetime64[ns]")


def require_panel_preserved(
    daily_panel: pd.DataFrame,
    classified: pd.DataFrame,
) -> None:
    """Compara identidad posicional del panel diario vs el clasificado.

    Permite exclusivamente la columna `sku_class`. No recalcula ADI/CV².
    No muta ninguno de los marcos.
    """
    faltan_diario = [c for c in CANONICAL_FIELDS if c not in daily_panel.columns]
    if faltan_diario:
        _fallar(f"el panel diario no expone {faltan_diario}")

    faltan_clasificado = [c for c in CANONICAL_FIELDS if c not in classified.columns]
    if faltan_clasificado:
        _fallar(f"el clasificado perdio columnas transaccionales {faltan_clasificado}")
    if TOPOLOGY_FIELD not in classified.columns:
        _fallar("falta sku_class en el panel clasificado")
    extras = [str(c) for c in classified.columns if str(c) not in PANEL_FIELDS]
    if extras:
        _fallar(f"columnas no permitidas en el clasificado: {extras}")

    n_antes = int(len(daily_panel))
    n_despues = int(len(classified))
    if n_antes != n_despues:
        _fallar(
            f"el numero de filas cambio de {n_antes} a {n_despues} "
            "(alta, baja o duplicacion indebida)"
        )

    antes = daily_panel.loc[:, list(CANONICAL_FIELDS)]
    despues = classified.loc[:, list(CANONICAL_FIELDS)]

    sku_antes = _como_texto(antes["sku_id"])
    sku_despues = _como_texto(despues["sku_id"])
    ts_antes = _como_timestamp_ns(antes["timestamp"])
    ts_despues = _como_timestamp_ns(despues["timestamp"])

    identidad_ok = np.array_equal(sku_antes, sku_despues) and np.array_equal(
        ts_antes, ts_despues
    )
    if not identidad_ok:
        pares_antes = set(zip(sku_antes.tolist(), ts_antes.tolist(), strict=True))
        pares_despues = set(zip(sku_despues.tolist(), ts_despues.tolist(), strict=True))
        if pares_antes != pares_despues:
            _fallar("se altero el conjunto de observaciones (sku_id, timestamp)")
        _fallar("se reordenaron filas respecto al panel diario")

    demanda_antes = _como_float(antes["demand_qty"])
    demanda_despues = _como_float(despues["demand_qty"])
    if not np.array_equal(demanda_antes, demanda_despues, equal_nan=True):
        _fallar("se muto demand_qty")

    lead_antes = _como_float(antes["lead_time_days"])
    lead_despues = _como_float(despues["lead_time_days"])
    if not np.array_equal(lead_antes, lead_despues, equal_nan=True):
        _fallar("se muto lead_time_days")

    _logger.info(
        "Preservacion 1.4 superada (filas=%s columnas_transaccionales_intactas=4)",
        n_antes,
    )
