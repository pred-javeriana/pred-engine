"""Exportador Parquet columnar hacia data/processed."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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
        # index=False conserva exactamente las columnas del marco de entrada.
        frame.to_parquet(
            destino,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )
        logger.info(
            "Exportacion Parquet completada: %s (%s filas)",
            destino,
            len(frame),
        )
        return destino
    except Exception:
        logger.exception("Fallo en la exportacion Parquet: %s", destino)
        raise
