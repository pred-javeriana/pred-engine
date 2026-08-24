"""Extractor CSV pasivo: lee crudo sin inferir tipos ni fechas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pred_engine.comun.logger import get_logger, log_ingestion_event
from pred_engine.ingesta.data import ensure_data_layout, raw_read_only_guard
from pred_engine.ingesta.lector.hashing import hash_sha256_archivo


@dataclass(frozen=True, slots=True)
class ExtractionArtifact:
    """Contrato inmutable de una extraccion pasiva."""

    frame: pd.DataFrame
    sha256: str
    row_count: int
    source_path: Path


def _assert_csv_bajo_raw(origen: Path, raw_root: Path) -> None:
    if origen.suffix.lower() != ".csv":
        raise ValueError("La extraccion pasiva solo admite archivos .csv")
    try:
        origen.expanduser().resolve().relative_to(raw_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"La extraccion pasiva solo lee desde {raw_root.resolve()}"
        ) from exc


def extract_csv(
    source: str | Path,
    *,
    data_root: str | Path | None = None,
) -> ExtractionArtifact:
    """Lee un CSV crudo y devuelve el marco mas el hash SHA-256 del archivo."""
    logger = get_logger(__name__)
    layout = ensure_data_layout(data_root)
    origen = Path(source)
    try:
        _assert_csv_bajo_raw(origen, layout.raw)
        with raw_read_only_guard(layout.raw):
            sha = hash_sha256_archivo(origen)
            # dtype=str evita inferencia de fechas y coercion numerica.
            marco = pd.read_csv(
                origen,
                dtype=str,
                parse_dates=False,
                keep_default_na=False,
                encoding="utf-8",
            )
        n_filas = int(len(marco))
        log_ingestion_event(
            logger,
            "Extraccion CSV pasiva completada",
            file_hash=sha,
            row_count=n_filas,
        )
        return ExtractionArtifact(
            frame=marco,
            sha256=sha,
            row_count=n_filas,
            source_path=origen.resolve(),
        )
    except Exception:
        logger.exception("Fallo en la extraccion CSV pasiva: %s", origen)
        raise
