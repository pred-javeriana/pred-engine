"""Composicion de sonda → barrera → remuestreo → topologia → contrato 1.4 → parquet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pred_engine.comun.llm import LlmProvider
from pred_engine.comun.logger import get_logger, log_ingestion_event
from pred_engine.ingesta.categorizacion import (
    TopologyArtifact,
    classify_daily_panel,
    select_canonical_columns,
)
from pred_engine.ingesta.continuidad import resample_daily
from pred_engine.ingesta.data import (
    DepositResult,
    OnConflict,
    deposit_raw,
    ensure_data_layout,
)
from pred_engine.ingesta.lector import (
    ExtractionArtifact,
    extract_csv,
    hash_sha256_archivo,
)
from pred_engine.ingesta.salida import (
    publish_classified_panel,
    read_classified_parquet,
    read_classified_parquet_with_sha256,
    require_positive_demand,
)
from pred_engine.ingesta.sonda import DiagnosticArtifact, probe_headers
from pred_engine.ingesta.validador_formato import validate_aligned_frame

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Artefacto de una corrida 1.2+1.3+1.4 (panel con sku_class publicado)."""

    source: ExtractionArtifact
    diagnostic: DiagnosticArtifact
    validated: pd.DataFrame
    panel: pd.DataFrame
    topology: TopologyArtifact
    parquet_path: Path
    deposit: DepositResult | None = None


@dataclass(frozen=True, slots=True)
class CsvIngestResult:
    """Artefacto de la ruta canonica (CSV con cabeceras canonicas, sin LLM)."""

    source_sha256: str
    row_count: int
    deposit: DepositResult
    validated_rows: int
    panel_rows: int
    topology: TopologyArtifact
    parquet_path: Path


def deposit_raw_csv(source: str | Path, *, data_root: str | Path | None = None) -> Path:
    """Copia el CSV del operador a raw/ y devuelve su ruta.

    Nunca sobrescribe: un archivo distinto con el mismo nombre lanza
    ``RawConflictError``. Para conservar ambos use ``deposit_raw``.
    """
    return deposit_raw(source, data_root=data_root, on_conflict="reject").path


def _clasificar_panel_diario(panel: pd.DataFrame) -> TopologyArtifact:
    """Gate de demanda positiva y clasificador real de 1.3 (sin stub)."""
    require_positive_demand(panel)
    return classify_daily_panel(panel)


def _publicar_handoff(
    panel_diario: pd.DataFrame,
    topologia: TopologyArtifact,
    destino: Path,
    *,
    data_root: Path,
    file_hash: str,
    mensaje: str,
) -> Path:
    """Preserva, valida el contrato 1.4 y persiste el Parquet."""
    escrito = publish_classified_panel(
        topologia.frame,
        destino,
        data_root=data_root,
        daily_panel=panel_diario,
    )
    log_ingestion_event(
        _logger,
        mensaje,
        file_hash=file_hash,
        row_count=int(len(topologia.frame)),
    )
    return escrito


def run_semantic_pipeline(
    frame: pd.DataFrame,
    provider: LlmProvider,
    *,
    timeout: float = 30.0,
) -> tuple[DiagnosticArtifact, pd.DataFrame, pd.DataFrame, TopologyArtifact]:
    """Puro respecto a filesystem: sonda → canonico → barrera → panel → topologia."""
    diagnostico = probe_headers(frame, provider, timeout=timeout)
    canonico = select_canonical_columns(diagnostico.frame)
    validado = validate_aligned_frame(canonico)
    panel = resample_daily(validado)
    topologia = _clasificar_panel_diario(panel)
    return diagnostico, validado, panel, topologia


def ingest_canonical_csv(
    csv_path: str | Path,
    *,
    data_root: str | Path | None = None,
    on_conflict: OnConflict = "reject",
) -> CsvIngestResult:
    """Deposita y clasifica un CSV con cabeceras canonicas (extras permitidas). Sin LLM.

    Deposita en raw/ igual que ``run_ingest`` y devuelve el hash de la fuente.
    """
    layout = ensure_data_layout(data_root)
    deposito = deposit_raw(csv_path, data_root=layout.root, on_conflict=on_conflict)
    marco = pd.read_csv(
        deposito.path,
        dtype=str,
        parse_dates=False,
        keep_default_na=False,
        encoding="utf-8",
    )
    canonico = select_canonical_columns(marco)
    validado = validate_aligned_frame(canonico)
    panel = resample_daily(validado)
    topologia = _clasificar_panel_diario(panel)
    destino = layout.processed / f"{deposito.path.stem}.parquet"
    _publicar_handoff(
        panel,
        topologia,
        destino,
        data_root=layout.root,
        file_hash=deposito.sha256,
        mensaje="Handoff 1.4 desde CSV completado",
    )
    return CsvIngestResult(
        source_sha256=deposito.sha256,
        row_count=len(marco),
        deposit=deposito,
        validated_rows=len(validado),
        panel_rows=len(topologia.frame),
        topology=topologia,
        parquet_path=destino,
    )


def run_classify_csv(
    csv_path: str | Path,
    *,
    data_root: str | Path | None = None,
) -> tuple[TopologyArtifact, Path]:
    """Clasifica un CSV con cabeceras canonicas (extras permitidas). Sin LLM."""
    resultado = ingest_canonical_csv(csv_path, data_root=data_root)
    return resultado.topology, resultado.parquet_path


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
    topologia = _clasificar_panel_diario(canonico)
    destino = layout.processed / f"{origen.stem}.parquet"
    _publicar_handoff(
        canonico,
        topologia,
        destino,
        data_root=layout.root,
        file_hash=hash_sha256_archivo(origen),
        mensaje="Handoff 1.4 desde Parquet completado",
    )
    return topologia, destino


def run_verify_parquet(parquet_path: str | Path) -> pd.DataFrame:
    """Relee un Parquet publicado y valida el contrato 1.4. No escribe."""
    return read_classified_parquet(parquet_path)


def run_verify_parquet_with_sha256(
    parquet_path: str | Path,
) -> tuple[pd.DataFrame, str]:
    """``run_verify_parquet`` con el SHA-256 de los bytes que se leyeron."""
    return read_classified_parquet_with_sha256(parquet_path)


def run_ingest(
    csv_path: str | Path,
    provider: LlmProvider,
    *,
    data_root: str | Path | None = None,
    timeout: float = 30.0,
    on_conflict: OnConflict = "reject",
) -> IngestResult:
    """Deposita, extrae (1.1), diagnostica (1.2), clasifica (1.3) y publica (1.4)."""
    layout = ensure_data_layout(data_root)
    deposito = deposit_raw(csv_path, data_root=layout.root, on_conflict=on_conflict)
    crudo = deposito.path
    extraido = extract_csv(crudo, data_root=layout.root)
    diagnostico, validado, panel, topologia = run_semantic_pipeline(
        extraido.frame,
        provider,
        timeout=timeout,
    )
    destino = layout.processed / f"{crudo.stem}.parquet"
    _publicar_handoff(
        panel,
        topologia,
        destino,
        data_root=layout.root,
        file_hash=extraido.sha256,
        mensaje="Pipeline 1.4 completado",
    )
    return IngestResult(
        source=extraido,
        diagnostic=diagnostico,
        validated=validado,
        panel=topologia.frame,
        topology=topologia,
        parquet_path=destino,
        deposit=deposito,
    )
