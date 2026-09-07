"""Composicion de sonda → proyeccion → barrera → remuestreo → topologia → parquet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copy2

import pandas as pd

from pred_engine.comun.llm import LlmProvider
from pred_engine.comun.logger import get_logger, log_ingestion_event
from pred_engine.ingesta.categorizacion import (
    TopologyArtifact,
    classify_panel,
    select_canonical_columns,
)
from pred_engine.ingesta.continuidad import resample_daily
from pred_engine.ingesta.data import ensure_data_layout
from pred_engine.ingesta.lector import (
    ExtractionArtifact,
    export_parquet,
    extract_csv,
    hash_sha256_archivo,
)
from pred_engine.ingesta.sonda import DiagnosticArtifact, probe_headers
from pred_engine.ingesta.validador_formato import validate_aligned_frame

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Artefacto de una corrida 1.2+1.3 (panel con sku_class)."""

    source: ExtractionArtifact
    diagnostic: DiagnosticArtifact
    validated: pd.DataFrame
    panel: pd.DataFrame
    topology: TopologyArtifact
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
) -> tuple[DiagnosticArtifact, pd.DataFrame, TopologyArtifact]:
    """Puro respecto a filesystem: sonda → canonico → barrera → panel → topologia."""
    diagnostico = probe_headers(frame, provider, timeout=timeout)
    canonico = select_canonical_columns(diagnostico.frame)
    validado = validate_aligned_frame(canonico)
    panel = resample_daily(validado)
    topologia = classify_panel(panel)
    return diagnostico, validado, topologia


def run_classify_csv(
    csv_path: str | Path,
    *,
    data_root: str | Path | None = None,
) -> tuple[TopologyArtifact, Path]:
    """Clasifica un CSV con cabeceras canonicas (extras permitidas). Sin LLM."""
    layout = ensure_data_layout(data_root)
    origen = Path(csv_path).expanduser().resolve()
    if not origen.is_file():
        raise FileNotFoundError(origen)
    marco = pd.read_csv(
        origen,
        dtype=str,
        parse_dates=False,
        keep_default_na=False,
        encoding="utf-8",
    )
    canonico = select_canonical_columns(marco)
    validado = validate_aligned_frame(canonico)
    panel = resample_daily(validado)
    topologia = classify_panel(panel)
    destino = layout.processed / f"{origen.stem}.parquet"
    export_parquet(topologia.frame, destino, data_root=layout.root)
    log_ingestion_event(
        _logger,
        "Clasificacion 1.3 desde CSV completada",
        file_hash=hash_sha256_archivo(origen),
        row_count=int(len(topologia.frame)),
    )
    return topologia, destino


def run_classify_parquet(
    parquet_path: str | Path,
    *,
    data_root: str | Path | None = None,
) -> tuple[TopologyArtifact, Path]:
    """Clasifica un panel Parquet 1.2 (columnas canonicas). Sin LLM."""
    layout = ensure_data_layout(data_root)
    origen = Path(parquet_path).expanduser().resolve()
    if not origen.is_file():
        raise FileNotFoundError(origen)
    marco = pd.read_parquet(origen, engine="pyarrow")
    canonico = select_canonical_columns(marco)
    topologia = classify_panel(canonico)
    destino = layout.processed / f"{origen.stem}.parquet"
    export_parquet(topologia.frame, destino, data_root=layout.root)
    log_ingestion_event(
        _logger,
        "Clasificacion 1.3 desde Parquet completada",
        file_hash=hash_sha256_archivo(origen),
        row_count=int(len(topologia.frame)),
    )
    return topologia, destino


def run_ingest(
    csv_path: str | Path,
    provider: LlmProvider,
    *,
    data_root: str | Path | None = None,
    timeout: float = 30.0,
) -> IngestResult:
    """Deposita, extrae (1.1), diagnostica/valida/remuestrea (1.2), clasifica (1.3)."""
    layout = ensure_data_layout(data_root)
    crudo = deposit_raw_csv(csv_path, data_root=layout.root)
    extraido = extract_csv(crudo, data_root=layout.root)
    diagnostico, validado, topologia = run_semantic_pipeline(
        extraido.frame,
        provider,
        timeout=timeout,
    )
    destino = layout.processed / f"{crudo.stem}.parquet"
    export_parquet(topologia.frame, destino, data_root=layout.root)
    log_ingestion_event(
        _logger,
        "Pipeline 1.3 completado",
        file_hash=extraido.sha256,
        row_count=int(len(topologia.frame)),
    )
    return IngestResult(
        source=extraido,
        diagnostic=diagnostico,
        validated=validado,
        panel=topologia.frame,
        topology=topologia,
        parquet_path=destino,
    )
