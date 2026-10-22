"""Barrera fail-fast: coercion explicita + Pydantic v2 por fila.

Las filas en el formato comun (fecha ISO, numeros decimales sin espacios ni
signos raros) se validan por columnas, de forma vectorizada. Toda fila que esa
via no puede garantizar pasa por la validacion fila a fila de siempre, en orden,
asi que el primer rechazo y su error son los mismos que con el bucle completo.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from typing import NoReturn

import numpy as np
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


def _validar_fila(
    sku_crudo: object,
    ts_crudo: object,
    demanda_cruda: object,
    lead_crudo: object,
    posicion: int,
) -> dict[str, object]:
    sku = str(sku_crudo).strip()
    try:
        observacion = InventoryObservation(
            sku_id=sku,
            timestamp=_parse_timestamp(ts_crudo, posicion),
            demand_qty=_parse_demand(demanda_cruda, posicion),
            lead_time_days=_parse_lead(lead_crudo, posicion),
        )
    except SchemaBarrierError:
        raise
    except ValidationError as exc:
        _fallar(
            f"violacion Pydantic en fila {posicion}: {exc}",
            row_index=posicion,
        )
    return observacion.model_dump()


def _tipar(filas: list[dict[str, object]]) -> pd.DataFrame:
    tipado = pd.DataFrame(filas)
    tipado["timestamp"] = pd.to_datetime(tipado["timestamp"]).astype("datetime64[ns]")
    tipado["demand_qty"] = tipado["demand_qty"].astype("float64")
    tipado["lead_time_days"] = tipado["lead_time_days"].astype("int64")
    tipado["sku_id"] = tipado["sku_id"].astype("string")
    return tipado


def _validar_por_filas(frame: pd.DataFrame) -> pd.DataFrame:
    filas = [
        _validar_fila(
            cruda["sku_id"],
            cruda["timestamp"],
            cruda["demand_qty"],
            cruda["lead_time_days"],
            posicion,
        )
        for posicion, (_, cruda) in enumerate(frame.iterrows())
    ]
    return _tipar(filas)


# Formatos que la via vectorizada sabe convertir igual que la validacion por fila.
# Solo digitos ASCII: ``float`` y ``to_datetime`` aceptan mas, y eso va por fila.
_FECHA_ISO = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}(?:[T ][0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?)?"
)
_DECIMAL = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
# Rango de datetime64[ns], el dtype del marco tipado.
_TS_MIN = pd.Timestamp("1678-01-01")
_TS_MAX = pd.Timestamp("2262-04-10")
# Enteros que float64 representa exactamente y caben en int64.
_LEAD_MAX = float(2**53)


def _por_valor_distinto(
    frame: pd.DataFrame,
    columna: str,
    evaluar: Callable[[pd.Series], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Evalua cada texto distinto una sola vez y expande el resultado a las filas.

    ``evaluar`` recibe los textos sin espacios (``str.strip`` de Python, igual que
    la validacion por fila) y devuelve el valor tipado y si la fila es segura.
    """
    codigos, distintos = pd.factorize(frame[columna])
    textos = pd.Series([str(texto).strip() for texto in distintos], dtype=object)
    valores, seguros = evaluar(textos)
    return valores[codigos], seguros[codigos]


def _skus(textos: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    return textos.to_numpy(dtype=object), (textos.str.len() > 0).to_numpy(
        dtype=bool, copy=True
    )


def _dias(textos: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    formato = textos.str.fullmatch(_FECHA_ISO).to_numpy(dtype=bool, copy=True)
    instantes = pd.to_datetime(textos.where(formato), format="ISO8601", errors="coerce")
    dias = instantes.dt.normalize()
    en_rango = dias.notna() & (dias >= _TS_MIN) & (dias <= _TS_MAX)
    return dias.to_numpy(), formato & en_rango.to_numpy(dtype=bool)


def _numeros(textos: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    formato = textos.str.fullmatch(_DECIMAL).to_numpy(dtype=bool, copy=True)
    valores = np.full(len(textos), np.nan)
    # Objeto -> float64 llama a ``float()`` por elemento: el mismo valor que por fila.
    valores[formato] = textos[formato].to_numpy(dtype=object).astype("float64")
    return valores, formato & np.isfinite(valores)


def _validar_vectorizado(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida por columnas; las filas dudosas se validan por fila y en orden."""
    skus, sku_ok = _por_valor_distinto(frame, "sku_id", _skus)
    dias, fecha_ok = _por_valor_distinto(frame, "timestamp", _dias)
    demanda, demanda_ok = _por_valor_distinto(frame, "demand_qty", _numeros)
    lead, lead_ok = _por_valor_distinto(frame, "lead_time_days", _numeros)
    demanda_ok &= demanda >= 0
    lead_ok &= (lead >= 1) & (lead <= _LEAD_MAX) & (np.floor(lead) == lead)
    seguras = sku_ok & fecha_ok & demanda_ok & lead_ok

    # La primera dudosa invalida lanza el mismo error que el bucle completo,
    # porque todas las filas anteriores son validas.
    validas = {
        int(posicion): _validar_fila(*frame.iloc[posicion], int(posicion))
        for posicion in np.flatnonzero(~seguras)
    }
    if validas:
        # Alguna fila valida no tiene el formato comun: se tipa todo como siempre.
        filas = [
            validas.get(posicion)
            or {
                "sku_id": skus[posicion],
                "timestamp": pd.Timestamp(dias[posicion]).to_pydatetime(),
                "demand_qty": float(demanda[posicion]),
                "lead_time_days": int(lead[posicion]),
            }
            for posicion in range(len(frame))
        ]
        return _tipar(filas)

    return pd.DataFrame(
        {
            "sku_id": pd.Series(skus, dtype="string"),
            "timestamp": pd.Series(dias).astype("datetime64[ns]"),
            "demand_qty": demanda.astype("float64"),
            "lead_time_days": lead.astype("int64"),
        }
    )


def validate_aligned_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida el marco post-sonda. Devuelve copia tipada; no muta el original."""
    if list(frame.columns) != list(CANONICAL_FIELDS):
        _fallar(
            "el marco alineado debe tener exactamente " + ", ".join(CANONICAL_FIELDS)
        )
    if frame.empty:
        _fallar("el marco alineado no tiene filas")

    # La via vectorizada solo conoce texto; cualquier otro valor va por fila.
    solo_texto = all(
        pd.api.types.infer_dtype(frame[columna], skipna=False) == "string"
        for columna in CANONICAL_FIELDS
    )
    tipado = _validar_vectorizado(frame) if solo_texto else _validar_por_filas(frame)

    duplicados = tipado.duplicated(subset=["sku_id", "timestamp"], keep=False)
    if bool(duplicados.any()):
        indice = int(duplicados.idxmax())
        _fallar(
            "filas duplicadas para (sku_id, timestamp)",
            row_index=indice,
        )
    _logger.info("Barrera de esquema superada (%s filas)", len(tipado))
    return tipado
