"""Exportador Parquet columnar hacia data/processed."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pred_engine.comun.logger import get_logger
from pred_engine.ingesta.data import (
    RawWritePermissionError,
    ensure_data_layout,
)


def _esta_bajo(ruta: Path, raiz: Path) -> bool:
    try:
        ruta.expanduser().resolve().relative_to(raiz.resolve())
        return True
    except (ValueError, OSError):
        return False


def export_parquet(
    frame: pd.DataFrame,
    destination: str | Path,
    *,
    data_root: str | Path | None = None,
    metadata: Mapping[Any, Any] | None = None,
) -> Path:
    """Escribe un DataFrame como Parquet comprimido con motor pyarrow."""
    logger = get_logger(__name__)
    layout = ensure_data_layout(data_root)
    destino = Path(destination)
    try:
        if destino.suffix.lower() != ".parquet":
            raise ValueError("El exportador solo escribe archivos .parquet")
        if _esta_bajo(destino, layout.raw):
            raise RawWritePermissionError(destino, "export_parquet")
        if not _esta_bajo(destino, layout.processed):
            raise ValueError(
                "El exportador solo escribe en "
                f"{layout.processed.resolve()}, no en {destino.resolve()}"
            )
        destino.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if metadata:
            meta_dict = dict(table.schema.metadata or {})
            for k, v in metadata.items():
                k_b = k.encode("utf-8") if isinstance(k, str) else k
                v_b = v.encode("utf-8") if isinstance(v, str) else v
                meta_dict[k_b] = v_b
            table = table.replace_schema_metadata(meta_dict)
        pq.write_table(table, destino, compression="snappy")
        logger.info(
            "Exportacion Parquet completada: %s (%s filas)",
            destino,
            len(frame),
        )
        return destino
    except Exception:
        logger.exception("Fallo en la exportacion Parquet: %s", destino)
        raise
