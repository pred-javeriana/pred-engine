"""Composicion de sonda → barrera → remuestreo → topologia → contrato 1.4 → parquet."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pred_engine.ingesta.informe import HeaderInfo, SourceFileInfo

import pandas as pd

from pred_engine.comun.exclusiones import sin_demanda_admisible
from pred_engine.comun.llm import LlmProvider
from pred_engine.comun.logger import get_logger, log_ingestion_event
from pred_engine.comun.reserva import ReserveCut
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
    """Gate de demanda positiva y clasificador real de 1.3 (sin stub).

    El corte sale del calendario completo, antes de excluir: quitar un SKU no
    mueve t* para los demas ni para el informe M1.
    """
    corte = ReserveCut.of(panel)
    excluidos = sin_demanda_admisible(panel, corte)
    admisible = panel.loc[~panel["sku_id"].isin([e.sku_id for e in excluidos])]
    require_positive_demand(panel if admisible.empty else admisible, corte)
    topologia = classify_daily_panel(admisible, corte)
    return replace(topologia, exclusions=excluidos)


def _publicar_handoff(
    panel_diario: pd.DataFrame,
    topologia: TopologyArtifact,
    destino: Path,
    *,
    data_root: Path,
    file_hash: str,
    mensaje: str,
    fuente: SourceFileInfo | None = None,
    deposito: DepositResult | None = None,
    validadas: int | None = None,
    cabecera: HeaderInfo | None = None,
) -> Path:
    """Preserva, valida el contrato 1.4 y persiste el Parquet."""
    escrito = publish_classified_panel(
        topologia.frame,
        destino,
        data_root=data_root,
        daily_panel=panel_diario.loc[
            ~panel_diario["sku_id"].isin([e.sku_id for e in topologia.exclusions])
        ],
    )
    log_ingestion_event(
        _logger,
        mensaje,
        file_hash=file_hash,
        row_count=int(len(topologia.frame)),
    )
    if fuente is not None:
        from pred_engine.ingesta.informe import (
            _informe_aceptado,
            write_ingest_report,
        )

        informe = _informe_aceptado(
            fuente,
            deposito,
            topologia,
            escrito,
            data_root,
            validadas=len(topologia.frame) if validadas is None else validadas,
            cabecera=cabecera,
        )
        write_ingest_report(informe, data_root)
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
    datos_sinteticos: bool | None = None,
    deposito: DepositResult | None = None,
) -> CsvIngestResult:
    """Deposita y clasifica un CSV con cabeceras canonicas (extras permitidas). Sin LLM.

    Deposita en raw/ igual que ``run_ingest`` y devuelve el hash de la fuente.
    """
    from pred_engine.ingesta.informe import SourceFileInfo, origen_informe

    layout = ensure_data_layout(data_root)
    if deposito is None:
        deposito = deposit_raw(csv_path, data_root=layout.root, on_conflict=on_conflict)
    origen_resuelto = origen_informe(layout.root, deposito.sha256, datos_sinteticos)
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
    fuente = SourceFileInfo(
        datos_sinteticos=origen_resuelto,
        name=deposito.path.name,
        sha256=deposito.sha256,
        row_count=len(marco),
    )
    _publicar_handoff(
        panel,
        topologia,
        destino,
        data_root=layout.root,
        file_hash=deposito.sha256,
        mensaje="Handoff 1.4 desde CSV completado",
        fuente=fuente,
        deposito=deposito,
        validadas=len(validado),
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
    datos_sinteticos: bool | None = None,
) -> tuple[TopologyArtifact, Path]:
    """Clasifica un CSV con cabeceras canonicas (extras permitidas). Sin LLM."""
    resultado = ingest_canonical_csv(
        csv_path, data_root=data_root, datos_sinteticos=datos_sinteticos
    )
    return resultado.topology, resultado.parquet_path


def run_classify_parquet(
    parquet_path: str | Path,
    *,
    data_root: str | Path | None = None,
    datos_sinteticos: bool | None = None,
) -> tuple[TopologyArtifact, Path]:
    """Clasifica un panel Parquet 1.2 (columnas canonicas). Sin LLM."""
    from pred_engine.ingesta.informe import SourceFileInfo, origen_informe

    layout = ensure_data_layout(data_root)
    origen = Path(parquet_path).expanduser().resolve()
    if not origen.is_file():
        raise FileNotFoundError(origen)
    sha256 = hash_sha256_archivo(origen)
    origen_resuelto = origen_informe(layout.root, sha256, datos_sinteticos)
    marco = pd.read_parquet(origen, engine="pyarrow")
    canonico = select_canonical_columns(marco)
    topologia = _clasificar_panel_diario(canonico)
    destino = layout.processed / f"{origen.stem}.parquet"
    fuente = SourceFileInfo(
        datos_sinteticos=origen_resuelto,
        name=origen.name,
        sha256=sha256,
        row_count=len(marco),
    )
    _publicar_handoff(
        canonico,
        topologia,
        destino,
        data_root=layout.root,
        file_hash=sha256,
        mensaje="Handoff 1.4 desde Parquet completado",
        fuente=fuente,
        validadas=len(marco),
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
    datos_sinteticos: bool | None = None,
    deposito: DepositResult | None = None,
) -> IngestResult:
    """Deposita, extrae (1.1), diagnostica (1.2), clasifica (1.3) y publica (1.4)."""
    from pred_engine.ingesta.informe import (
        SourceFileInfo,
        _cabecera,
        origen_informe,
    )

    layout = ensure_data_layout(data_root)
    if deposito is None:
        deposito = deposit_raw(csv_path, data_root=layout.root, on_conflict=on_conflict)
    crudo = deposito.path
    extraido = extract_csv(crudo, data_root=layout.root)
    origen_resuelto = origen_informe(layout.root, extraido.sha256, datos_sinteticos)
    diagnostico, validado, panel, topologia = run_semantic_pipeline(
        extraido.frame,
        provider,
        timeout=timeout,
    )
    destino = layout.processed / f"{crudo.stem}.parquet"
    fuente = SourceFileInfo(
        datos_sinteticos=origen_resuelto,
        name=deposito.path.name,
        sha256=extraido.sha256,
        row_count=extraido.row_count,
    )
    _publicar_handoff(
        panel,
        topologia,
        destino,
        data_root=layout.root,
        file_hash=extraido.sha256,
        mensaje="Pipeline 1.4 completado",
        fuente=fuente,
        deposito=deposito,
        validadas=len(validado),
        cabecera=_cabecera(diagnostico.diagnostic),
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
