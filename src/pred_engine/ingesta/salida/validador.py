"""Validador fail-closed del contrato final de cinco columnas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, NoReturn, cast

import pandas as pd
from pydantic import ValidationError

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos import (
    PANEL_FIELDS,
    SKU_CLASSES,
    ClassifiedObservation,
    SkuClass,
)
from pred_engine.ingesta.salida.errores import OutputContractError

_logger = get_logger(__name__)


def _fallar(mensaje: str, *, column: str | None = None) -> NoReturn:
    # No se registran valores crudos: el handoff no debe filtrar demanda ni PII.
    _logger.error("Contrato 1.4 violado: %s (columna=%s)", mensaje, column)
    raise OutputContractError(mensaje)


def _timestamp_python(crudo: object) -> datetime:
    ts = pd.Timestamp(cast(Any, crudo))
    if pd.isna(ts):
        raise ValueError("timestamp nulo")
    python_dt = cast(datetime, ts.to_pydatetime())
    if python_dt.tzinfo is not None:
        python_dt = python_dt.replace(tzinfo=None)
    return python_dt.replace(microsecond=0)


def _contar_skus(serie: pd.Series) -> int:
    return len({str(v) for v in serie.tolist() if pd.notna(v)})


def normalize_output_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Copia ordenada con los dtypes del contrato. No muta el origen."""
    copia = frame.loc[:, list(PANEL_FIELDS)].copy()
    copia["sku_id"] = copia["sku_id"].astype("string")
    copia["timestamp"] = pd.to_datetime(copia["timestamp"], utc=False).astype(
        "datetime64[ns]"
    )
    copia["demand_qty"] = copia["demand_qty"].astype("float64")
    copia["lead_time_days"] = copia["lead_time_days"].astype("int64")
    copia["sku_class"] = copia["sku_class"].astype("string")
    return copia.reset_index(drop=True)


def validate_output_contract(frame: pd.DataFrame) -> None:
    """Rechaza el panel completo si viola el contrato 1.4. No muta `frame`."""
    if list(frame.columns) != list(PANEL_FIELDS):
        _fallar(
            "el marco debe contener exactamente " + ", ".join(PANEL_FIELDS),
        )
    if frame.empty:
        _fallar("el panel clasificado no tiene filas")

    for columna in PANEL_FIELDS:
        if bool(frame[columna].isna().any()):
            _fallar(
                f"valores nulos en columna obligatoria {columna}",
                column=columna,
            )

    duplicados = frame.duplicated(subset=["sku_id", "timestamp"], keep=False)
    if bool(duplicados.any()):
        _fallar("filas duplicadas para (sku_id, timestamp)")

    sku_col = cast(pd.Series, frame["sku_id"])
    if not (
        pd.api.types.is_string_dtype(sku_col) or pd.api.types.is_object_dtype(sku_col)
    ):
        _fallar("sku_id debe ser String", column="sku_id")
    sku_texto = sku_col.astype("string")
    if bool(sku_texto.str.strip().eq("").any()):
        _fallar("sku_id no puede ser vacio", column="sku_id")

    if not pd.api.types.is_datetime64_any_dtype(frame["timestamp"]):
        _fallar("timestamp debe ser Datetime64", column="timestamp")

    demanda = frame["demand_qty"]
    if not pd.api.types.is_float_dtype(demanda):
        _fallar("demand_qty debe ser Float", column="demand_qty")
    if bool((demanda < 0).any()):
        _fallar("demand_qty contiene valores negativos", column="demand_qty")

    lead = frame["lead_time_days"]
    if not pd.api.types.is_integer_dtype(lead):
        _fallar("lead_time_days debe ser Integer", column="lead_time_days")
    if bool((lead < 1).any()):
        _fallar("lead_time_days no respeta el dominio >= 1", column="lead_time_days")

    clases = frame["sku_class"].astype("string")
    permitidas = set(SKU_CLASSES)
    invalidas = sorted(set(clases.dropna()) - permitidas)
    if invalidas:
        _fallar(
            "sku_class admite solo smooth, intermittent, erratic, lumpy",
            column="sku_class",
        )

    n_etiquetas = frame.groupby("sku_id", sort=False)["sku_class"].nunique(dropna=False)
    if not bool((n_etiquetas == 1).all()):
        _fallar("sku_class no es constante dentro de cada sku_id", column="sku_class")

    for posicion, (_, cruda) in enumerate(frame.iterrows()):
        try:
            ClassifiedObservation(
                sku_id=str(cruda["sku_id"]).strip(),
                timestamp=_timestamp_python(cruda["timestamp"]),
                demand_qty=float(cast(Any, cruda["demand_qty"])),
                lead_time_days=int(cast(Any, cruda["lead_time_days"])),
                sku_class=cast(SkuClass, str(cruda["sku_class"])),
            )
        except (TypeError, ValueError, ValidationError):
            _fallar(f"violacion Pydantic en fila {posicion}")

    _logger.info(
        "Contrato 1.4 superado (filas=%s skus=%s)",
        int(len(frame)),
        _contar_skus(cast(pd.Series, frame["sku_id"])),
    )
