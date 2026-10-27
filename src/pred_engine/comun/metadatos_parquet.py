"""Metadatos de corte y exclusiones persistidos en el schema del archivo Parquet.

Permite que cada paso posterior a la ingesta (clasificacion desde Parquet, L2,
resumen, topologia, M3) consulte t* y las exclusiones directamente del archivo,
sin recalcular t* ni depender de informes externos.
"""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import pyarrow.parquet as pq

from pred_engine.comun.errores_handoff import MissingCutoffMetadataError
from pred_engine.comun.exclusiones import SkuExclusion
from pred_engine.comun.reserva import RESERVE_FRACTION, ReserveCut

METADATA_KEY: Final = b"pred_engine"
ERROR_MENSAJE_REIMPORTAR: Final = (
    "el archivo Parquet no contiene los metadatos requeridos de pred_engine; "
    "vuelva a importar desde el CSV (re-import from the CSV)"
)


@dataclass(frozen=True, slots=True)
class ParquetMetadata:
    """Metadatos almacenados bajo la clave pred_engine en el schema Parquet."""

    t_star: pd.Timestamp
    first_reserved: pd.Timestamp
    last_observed: pd.Timestamp
    reserved_days: int
    fraction: float = RESERVE_FRACTION
    exclusions: tuple[SkuExclusion, ...] = ()

    @property
    def reserve(self) -> ReserveCut:
        return ReserveCut(
            t_star=self.t_star,
            first_reserved=self.first_reserved,
            last_observed=self.last_observed,
            reserved_days=self.reserved_days,
            fraction=self.fraction,
        )


def serializar_metadatos(metadata: ParquetMetadata) -> bytes:
    """Serializa la metadata a bytes JSON UTF-8."""
    payload = {
        "t_star": metadata.t_star.date().isoformat(),
        "first_reserved": metadata.first_reserved.date().isoformat(),
        "last_observed": metadata.last_observed.date().isoformat(),
        "reserved_days": int(metadata.reserved_days),
        "fraction": float(metadata.fraction),
        "exclusions": [
            {"sku_id": str(e.sku_id), "cause": str(e.cause)}
            for e in metadata.exclusions
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


def deserializar_metadatos(datos_bytes: bytes) -> ParquetMetadata:
    """Reconstruye ParquetMetadata desde bytes JSON UTF-8."""
    try:
        payload = json.loads(datos_bytes.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("payload invalido")
        t_star = pd.Timestamp(payload["t_star"])
        first_reserved = pd.Timestamp(payload["first_reserved"])
        last_observed = pd.Timestamp(payload["last_observed"])
        if not (
            isinstance(t_star, pd.Timestamp)
            and isinstance(first_reserved, pd.Timestamp)
            and isinstance(last_observed, pd.Timestamp)
        ):
            raise ValueError("fechas invalidas en metadatos")
        return ParquetMetadata(
            t_star=t_star.normalize(),
            first_reserved=first_reserved.normalize(),
            last_observed=last_observed.normalize(),
            reserved_days=int(payload["reserved_days"]),
            fraction=float(payload.get("fraction", RESERVE_FRACTION)),
            exclusions=tuple(
                SkuExclusion(sku_id=str(item["sku_id"]), cause=str(item["cause"]))
                for item in payload.get("exclusions", [])
            ),
        )
    except Exception as exc:
        raise MissingCutoffMetadataError(ERROR_MENSAJE_REIMPORTAR) from exc


def empaquetar_metadatos_schema(
    reserve: ReserveCut,
    exclusions: Sequence[SkuExclusion] = (),
) -> dict[bytes, bytes]:
    """Genera el diccionario de metadatos de schema para pyarrow."""
    metadata = ParquetMetadata(
        t_star=reserve.t_star,
        first_reserved=reserve.first_reserved,
        last_observed=reserve.last_observed,
        reserved_days=reserve.reserved_days,
        fraction=reserve.fraction,
        exclusions=tuple(exclusions),
    )
    return {METADATA_KEY: serializar_metadatos(metadata)}


def leer_metadatos_parquet(
    source: str | Path | io.BytesIO | bytes,
) -> ParquetMetadata:
    """Lee y valida los metadatos de pred_engine en un archivo Parquet.

    Si no estan presentes o son invalidos, lanza MissingCutoffMetadataError.
    """
    try:
        if isinstance(source, bytes):
            pf = pq.ParquetFile(io.BytesIO(source))
        elif isinstance(source, io.BytesIO):
            pf = pq.ParquetFile(source)
        else:
            path = Path(source).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            pf = pq.ParquetFile(path)
    except MissingCutoffMetadataError:
        raise
    except Exception as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        raise MissingCutoffMetadataError(ERROR_MENSAJE_REIMPORTAR) from exc

    meta = pf.schema_arrow.metadata or {}
    raw = meta.get(METADATA_KEY) or meta.get(METADATA_KEY.decode("utf-8"))
    if raw is None:
        raise MissingCutoffMetadataError(ERROR_MENSAJE_REIMPORTAR)

    raw_bytes = raw if isinstance(raw, bytes) else raw.encode("utf-8")
    return deserializar_metadatos(raw_bytes)
