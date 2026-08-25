"""Composicion de sonda → barrera → remuestreo → parquet (sin estado de modulo)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copy2

import pandas as pd

from pred_engine.comun.llm import LlmProvider
from pred_engine.comun.logger import get_logger, log_ingestion_event
from pred_engine.ingesta.continuidad import resample_daily
from pred_engine.ingesta.data import ensure_data_layout
from pred_engine.ingesta.lector import ExtractionArtifact, export_parquet, extract_csv
from pred_engine.ingesta.sonda import DiagnosticArtifact, probe_headers
from pred_engine.ingesta.validador_formato import validate_aligned_frame

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Artefacto de una corrida completa de alineacion 1.2."""

    source: ExtractionArtifact
    diagnostic: DiagnosticArtifact
    validated: pd.DataFrame
    panel: pd.DataFrame
    parquet_path: Path


def deposit_raw_csv(source: str | Path, *, data_root: str | Path | None = None) -> Path:
    """Copia el CSV del operador a raw/. No usa la guardia (deposito, no pipeline)."""
    layout = ensure_data_layout(data_root)
    origen = Path(source).expanduser().resolve()
    if not origen.is_file():
        raise FileNotFoundError(origen)
    destino = (layout.raw / origen.name).resolve()
    if origen != destino:
        copy2(origen, destino)
    return destino


def run_semantic_pipeline(
    frame: pd.DataFrame,
    provider: LlmProvider,
    *,
    timeout: float = 30.0,
) -> tuple[DiagnosticArtifact, pd.DataFrame, pd.DataFrame]:
    """Puro respecto a filesystem: sonda consultiva → barrera → panel diario."""
    diagnostico = probe_headers(frame, provider, timeout=timeout)
    validado = validate_aligned_frame(diagnostico.frame)
    panel = resample_daily(validado)
    return diagnostico, validado, panel


def run_ingest(
    csv_path: str | Path,
    provider: LlmProvider,
    *,
    data_root: str | Path | None = None,
    timeout: float = 30.0,
) -> IngestResult:
    """Deposita, extrae (1.1), diagnostica/valida/remuestrea (1.2) y exporta Parquet."""
    layout = ensure_data_layout(data_root)
    crudo = deposit_raw_csv(csv_path, data_root=layout.root)
    extraido = extract_csv(crudo, data_root=layout.root)
    diagnostico, validado, panel = run_semantic_pipeline(
        extraido.frame,
        provider,
        timeout=timeout,
    )
    destino = layout.processed / f"{crudo.stem}.parquet"
    export_parquet(panel, destino, data_root=layout.root)
    log_ingestion_event(
        _logger,
        "Pipeline 1.2 completado",
        file_hash=extraido.sha256,
        row_count=int(len(panel)),
    )
    return IngestResult(
        source=extraido,
        diagnostic=diagnostico,
        validated=validado,
        panel=panel,
        parquet_path=destino,
    )
