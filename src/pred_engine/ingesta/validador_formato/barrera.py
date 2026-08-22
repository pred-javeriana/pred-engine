"""Barrera fail-fast: coercion explicita + Pydantic v2 por fila."""

from __future__ import annotations

from datetime import datetime
from typing import NoReturn

import pandas as pd
from pydantic import ValidationError

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos import CANONICAL_FIELDS, InventoryObservation
from pred_engine.ingesta.validador_formato.errores import SchemaBarrierError

_logger = get_logger(__name__)


def _fallar(
    mensaje: str,
    *,
    row_index: int | None = None,
    column: str | None = None,
    raw_value: object = None,
) -> NoReturn:
    _logger.error(
        "Barrera de esquema: %s (fila=%s, columna=%s, crudo=%r)",
        mensaje,
        row_index,
        column,
        raw_value,
    )
    raise SchemaBarrierError(
        mensaje,
        row_index=row_index,
        column=column,
        raw_value=raw_value,
    )


def _parse_timestamp(crudo: object, fila: int) -> datetime:
    texto = str(crudo).strip()
    if not texto:
        _fallar("timestamp vacio", row_index=fila, column="timestamp", raw_value=crudo)
    try:
        ts = pd.to_datetime(texto, format="ISO8601", utc=False)
    except (ValueError, TypeError):
        _fallar(
            f"timestamp invalido: {texto!r}",
            row_index=fila,
            column="timestamp",
            raw_value=crudo,
        )
    if pd.isna(ts):
        _fallar(
            f"timestamp invalido: {texto!r}",
            row_index=fila,
            column="timestamp",
            raw_value=crudo,
        )
    python_dt = pd.Timestamp(ts).to_pydatetime()
    return python_dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _parse_demand(crudo: object, fila: int) -> float:
    texto = str(crudo).strip()
    try:
        valor = float(texto)
    except (TypeError, ValueError):
        _fallar(
            f"demand_qty no numerica: {texto!r}",
            row_index=fila,
            column="demand_qty",
            raw_value=crudo,
        )
    if valor < 0:
        _fallar(
            f"demand_qty negativa: {valor}",
            row_index=fila,
            column="demand_qty",
            raw_value=crudo,
        )
    return float(valor)


def _parse_lead(crudo: object, fila: int) -> int:
    texto = str(crudo).strip()
    try:
        valor = float(texto)
    except (TypeError, ValueError):
        _fallar(
            f"lead_time_days no numerico: {texto!r}",
            row_index=fila,
            column="lead_time_days",
            raw_value=crudo,
        )
    if not valor.is_integer():
        _fallar(
            f"lead_time_days no entero: {texto!r}",
            row_index=fila,
            column="lead_time_days",
            raw_value=crudo,
        )
    entero = int(valor)
    if entero < 1:
        _fallar(
            f"lead_time_days < 1: {entero}",
            row_index=fila,
            column="lead_time_days",
            raw_value=crudo,
        )
    return entero


def validate_aligned_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida el marco post-sonda. Devuelve copia tipada; no muta el original."""
    if list(frame.columns) != list(CANONICAL_FIELDS):
        _fallar(
            "el marco alineado debe tener exactamente " + ", ".join(CANONICAL_FIELDS)
        )
    if frame.empty:
        _fallar("el marco alineado no tiene filas")

    filas: list[dict[str, object]] = []
    for posicion, (_, cruda) in enumerate(frame.iterrows()):
        sku = str(cruda["sku_id"]).strip()
        try:
            observacion = InventoryObservation(
                sku_id=sku,
                timestamp=_parse_timestamp(cruda["timestamp"], posicion),
                demand_qty=_parse_demand(cruda["demand_qty"], posicion),
                lead_time_days=_parse_lead(cruda["lead_time_days"], posicion),
            )
        except SchemaBarrierError:
            raise
        except ValidationError as exc:
            _fallar(
                f"violacion Pydantic en fila {posicion}: {exc}",
                row_index=posicion,
            )
        filas.append(observacion.model_dump())

    tipado = pd.DataFrame(filas)
    tipado["timestamp"] = pd.to_datetime(tipado["timestamp"]).astype("datetime64[ns]")
    tipado["demand_qty"] = tipado["demand_qty"].astype("float64")
    tipado["lead_time_days"] = tipado["lead_time_days"].astype("int64")
    tipado["sku_id"] = tipado["sku_id"].astype("string")

    duplicados = tipado.duplicated(subset=["sku_id", "timestamp"], keep=False)
    if bool(duplicados.any()):
        indice = int(duplicados.idxmax())
        _fallar(
            "filas duplicadas para (sku_id, timestamp)",
            row_index=indice,
        )
    _logger.info("Barrera de esquema superada (%s filas)", len(tipado))
    return tipado
