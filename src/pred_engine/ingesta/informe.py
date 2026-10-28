"""Informe de ingesta en archivo: lo que la plataforma importa a su DAL (G1).

Cada ingesta deja ``<data_root>/ingestas/<sha256>.json`` (esquema
``pred-engine.ingesta/1``, descrito en la seccion 17 del contrato frontend-motor),
tambien cuando termina rechazada, fallida o a la espera de confirmacion. La
plataforma lo lee; el motor nunca depende de ella.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pred_engine.comun.errores_codigo import codigo_de
from pred_engine.comun.llm.errores import LlmProviderError, LlmTimeoutError
from pred_engine.comun.llm.protocolo import LlmProvider
from pred_engine.comun.modelos import (
    ADI_THRESHOLD,
    CV2_THRESHOLD,
    DiagnosticEntry,
    HeaderDiagnostic,
    TopologyMetrics,
)
from pred_engine.comun.procedencia import (
    OrigenIncompatibleError,
    origen_ingesta,
    resolver_origen,
)
from pred_engine.comun.reserva import ReserveCut
from pred_engine.ingesta.categorizacion import (
    TopologyArtifact,
)
from pred_engine.ingesta.data import (
    DepositResult,
    OnConflict,
    RawConflictError,
    deposit_raw,
    ensure_data_layout,
)
from pred_engine.ingesta.lector.hashing import hash_sha256_archivo
from pred_engine.ingesta.pipeline import (
    CsvIngestResult,
    IngestResult,
    ingest_canonical_csv,
    run_ingest,
)
from pred_engine.ingesta.sonda.errores import SemanticAlignmentError
from pred_engine.ingesta.validador_formato.errores import SchemaBarrierError

REPORT_SCHEMA = "pred-engine.ingesta/1"
REPORTS_DIR = "ingestas"

IngestStatus = Literal["accepted", "rejected", "failed", "needs_confirmation"]
_ETAPA = "L1"


class _Modelo(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SourceFileInfo(_Modelo):
    datos_sinteticos: bool | None = Field(default=None, strict=True)
    name: str
    sha256: str
    row_count: int = Field(ge=0)


class DepositInfo(_Modelo):
    raw_path: str
    status: Literal[
        "deposited", "already_present_identical", "already_present_different"
    ]
    would_overwrite: bool


class HeaderInfo(_Modelo):
    status: Literal["accepted", "rejected"]
    entries: list[DiagnosticEntry] = Field(default_factory=list)


class ValidationInfo(_Modelo):
    rows_validated: int = Field(ge=0)
    rows_daily_panel: int = Field(ge=0)
    n_skus: int = Field(ge=0)


class PublishedInfo(_Modelo):
    sha256: str | None = None
    parquet_path: str
    rows: int = Field(ge=0)
    n_skus: int = Field(ge=0)
    columns: list[str]


class QualityEntryInfo(_Modelo):
    severity: Literal["info", "warning", "error"]
    code: str
    message: str
    row_index: int | None = None
    column: str | None = None


class ErrorDetail(_Modelo):
    code: str
    stage: str = _ETAPA
    detail: str
    field: str | None = None
    row_index: int | None = None
    column: str | None = None
    source_exception: str
    retryable: bool = False


class ReserveInfo(_Modelo):
    t_star: str
    first_reserved: str
    last_observed: str
    reserved_days: int = Field(ge=1)
    fraction: float


class ThresholdsInfo(_Modelo):
    adi: float
    cv2: float


class TopologyInfo(_Modelo):
    thresholds: ThresholdsInfo
    metrics: list[TopologyMetrics]


class ExclusionInfo(_Modelo):
    sku_id: str
    cause: str


class IngestReportFile(_Modelo):
    """Contenido de ``ingestas/<sha256>.json``."""

    schema_id: str = Field(default=REPORT_SCHEMA, alias="schema")
    written_at: str
    ingest_id: str
    status: IngestStatus
    source_file: SourceFileInfo
    deposit: DepositInfo | None = None
    header_diagnostic: HeaderInfo | None = None
    validation: ValidationInfo | None = None
    published: PublishedInfo | None = None
    quality_log: list[QualityEntryInfo] = Field(default_factory=list)
    error: ErrorDetail | None = None
    reserve: ReserveInfo | None = None
    topology: TopologyInfo | None = None
    excluded_count: int = 0
    exclusions: list[ExclusionInfo] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ReportedIngest:
    """Resultado de ``ingest_csv_with_report``: el informe y lo que lo origino."""

    report: IngestReportFile
    path: Path
    result: CsvIngestResult | IngestResult | None
    error: Exception | None


def error_code(exc: BaseException) -> str:
    """Codigo estable para ``exc`` (ver seccion 11 del contrato)."""
    return codigo_de(exc)


def report_path(data_root: str | Path | None, sha256: str) -> Path:
    """Ruta del informe de la ingesta cuyo CSV tiene ese SHA-256."""
    return ensure_data_layout(data_root).ingestas / f"{sha256}.json"


def _relativa(ruta: Path, raiz: Path) -> str:
    try:
        return ruta.resolve().relative_to(raiz.resolve()).as_posix()
    except ValueError:
        return ruta.resolve().as_posix()


def _contar_filas(ruta: Path) -> int:
    with ruta.open("rb") as archivo:
        lineas = sum(1 for _ in archivo)
    return max(lineas - 1, 0)


def _deposito_info(deposito: DepositResult, raiz: Path) -> DepositInfo:
    return DepositInfo(
        raw_path=_relativa(deposito.path, raiz),
        status=deposito.status,
        would_overwrite=deposito.would_overwrite,
    )


def _topologia(artefacto: TopologyArtifact) -> tuple[ReserveInfo, TopologyInfo]:
    corte = (
        artefacto.reserve
        if artefacto.reserve is not None
        else ReserveCut.of(artefacto.frame)
    )
    reserva = ReserveInfo(
        t_star=corte.t_star.date().isoformat(),
        first_reserved=corte.first_reserved.date().isoformat(),
        last_observed=corte.last_observed.date().isoformat(),
        reserved_days=corte.reserved_days,
        fraction=corte.fraction,
    )
    topologia = TopologyInfo(
        thresholds=ThresholdsInfo(adi=ADI_THRESHOLD, cv2=CV2_THRESHOLD),
        metrics=list(artefacto.metrics),
    )
    return reserva, topologia


def _ahora() -> str:
    return datetime.now(UTC).isoformat()


def _cabecera(diagnostico: HeaderDiagnostic | None) -> HeaderInfo | None:
    if diagnostico is None:
        return None
    return HeaderInfo(status=diagnostico.status, entries=list(diagnostico.diagnostic))


def _informe_aceptado(
    fuente: SourceFileInfo,
    deposito: DepositResult,
    resultado: CsvIngestResult | IngestResult,
    raiz: Path,
) -> IngestReportFile:
    topologia = resultado.topology
    reserva, info = _topologia(topologia)
    if isinstance(resultado, IngestResult):
        cabecera = _cabecera(resultado.diagnostic.diagnostic)
        validadas = len(resultado.validated)
    else:
        cabecera = None
        validadas = resultado.validated_rows
    n_skus = len(topologia.metrics)
    exclusiones = [
        ExclusionInfo(sku_id=e.sku_id, cause=e.cause) for e in topologia.exclusions
    ]
    return IngestReportFile(
        written_at=_ahora(),
        ingest_id=fuente.sha256,
        status="accepted",
        source_file=fuente,
        deposit=_deposito_info(deposito, raiz),
        header_diagnostic=cabecera,
        validation=ValidationInfo(
            rows_validated=validadas,
            rows_daily_panel=len(topologia.frame),
            n_skus=n_skus,
        ),
        published=PublishedInfo(
            sha256=hash_sha256_archivo(resultado.parquet_path),
            parquet_path=_relativa(resultado.parquet_path, raiz),
            rows=len(topologia.frame),
            n_skus=n_skus,
            columns=[str(c) for c in topologia.frame.columns],
        ),
        reserve=reserva,
        topology=info,
        excluded_count=len(exclusiones),
        exclusions=exclusiones,
    )


def _informe_fallido(
    fuente: SourceFileInfo,
    deposito: DepositResult | None,
    exc: Exception,
    raiz: Path,
) -> IngestReportFile:
    codigo = error_code(exc)
    fila = exc.row_index if isinstance(exc, SchemaBarrierError) else None
    columna = exc.column if isinstance(exc, SchemaBarrierError) else None
    detalle = ErrorDetail(
        code=codigo,
        detail=str(exc),
        row_index=fila,
        column=columna,
        source_exception=type(exc).__name__,
        retryable=isinstance(exc, LlmTimeoutError | LlmProviderError),
    )
    cabecera: HeaderInfo | None = None
    registro: list[QualityEntryInfo] = []
    estado: IngestStatus = "failed"
    if isinstance(exc, OrigenIncompatibleError):
        estado = "rejected"
    if isinstance(exc, SemanticAlignmentError):
        estado = "rejected"
        cabecera = _cabecera(exc.diagnostic) or HeaderInfo(status="rejected")
        registro = [
            QualityEntryInfo(
                severity="error" if entrada.severity == "error" else "info",
                code=codigo,
                message=entrada.message,
                column=entrada.field,
            )
            for entrada in cabecera.entries
        ] or [QualityEntryInfo(severity="error", code=codigo, message=str(exc))]
    else:
        registro = [
            QualityEntryInfo(
                severity="error",
                code=codigo,
                message=str(exc),
                row_index=fila,
                column=columna,
            )
        ]
    return IngestReportFile(
        written_at=_ahora(),
        ingest_id=fuente.sha256,
        status=estado,
        source_file=fuente,
        deposit=None if deposito is None else _deposito_info(deposito, raiz),
        header_diagnostic=cabecera,
        quality_log=registro,
        error=detalle,
    )


def _informe_conflicto(
    fuente: SourceFileInfo, exc: RawConflictError, raiz: Path
) -> IngestReportFile:
    return IngestReportFile(
        written_at=_ahora(),
        ingest_id=fuente.sha256,
        status="needs_confirmation",
        source_file=fuente,
        deposit=DepositInfo(
            raw_path=_relativa(exc.existing, raiz),
            status="already_present_different",
            would_overwrite=True,
        ),
        quality_log=[
            QualityEntryInfo(
                severity="warning",
                code=error_code(exc),
                message=str(exc),
            )
        ],
    )


def origen_informe(
    data_root: str | Path | None, sha256: str, declarada: bool | None = None
) -> bool | None:
    """Lee solo el informe canonico de esa fuente, sin deducir un origen ausente."""
    path = report_path(data_root, sha256)
    conocida = None
    if path.exists():
        previa = IngestReportFile.model_validate_json(path.read_text())
        if previa.ingest_id != sha256 or previa.source_file.sha256 != sha256:
            raise ValueError("la procedencia M1 tiene identidades incompatibles")
        conocida = previa.source_file.datos_sinteticos
    return resolver_origen(conocida, declarada)


def write_ingest_report(report: IngestReportFile, data_root: str | Path | None) -> Path:
    """Escribe el informe de forma atomica y devuelve su ruta."""
    origen = origen_informe(
        data_root, report.ingest_id, report.source_file.datos_sinteticos
    )
    if origen != report.source_file.datos_sinteticos:
        report = report.model_copy(
            update={
                "source_file": report.source_file.model_copy(
                    update={"datos_sinteticos": origen}
                )
            }
        )
    destino = report_path(data_root, report.ingest_id)
    temporal = destino.with_name(f".{destino.name}.{os.getpid()}.part")
    try:
        temporal.write_text(
            report.model_dump_json(indent=2, by_alias=True) + "\n", encoding="utf-8"
        )
        os.replace(temporal, destino)
    finally:
        temporal.unlink(missing_ok=True)
    return destino


def ingest_csv_with_report(
    csv_path: str | Path,
    *,
    data_root: str | Path | None = None,
    provider: LlmProvider | None = None,
    timeout: float = 30.0,
    on_conflict: OnConflict = "reject",
    datos_sinteticos: bool | None = None,
) -> ReportedIngest:
    """Ingiere un CSV y deja siempre su informe en ``ingestas/<sha256>.json``.

    Sin ``provider`` usa la ruta canonica (sin LLM). Los errores de la ingesta no
    se propagan: quedan en ``error`` y en el informe, para que el llamador decida
    como terminar. Solo ``FileNotFoundError`` del origen se propaga, porque sin
    archivo no hay hash con el que nombrar el informe.
    """
    layout = ensure_data_layout(data_root)
    origen = Path(csv_path).expanduser().resolve()
    if not origen.is_file():
        raise FileNotFoundError(origen)
    sha256 = hash_sha256_archivo(origen)
    fuente = SourceFileInfo(
        datos_sinteticos=origen_informe(layout.root, sha256, datos_sinteticos),
        name=origen.name,
        sha256=sha256,
        row_count=_contar_filas(origen),
    )

    deposito: DepositResult | None = None
    resultado: CsvIngestResult | IngestResult | None = None
    error: Exception | None = None
    try:
        deposito = deposit_raw(origen, data_root=layout.root, on_conflict=on_conflict)
        # Ya esta en raw/: las rutas de ingesta lo reconocen sin copiarlo de nuevo.
        if provider is None:
            resultado = ingest_canonical_csv(deposito.path, data_root=layout.root)
        else:
            resultado = run_ingest(
                deposito.path, provider, data_root=layout.root, timeout=timeout
            )
    except Exception as exc:  # noqa: BLE001 - se registra en el informe y se devuelve
        error = exc

    if error is None and resultado is not None and deposito is not None:
        informe = _informe_aceptado(fuente, deposito, resultado, layout.root)
        assert informe.published is not None
        try:
            # Los mismos bytes publicados por otra ingesta no cambian de origen.
            origen_ingesta(
                None,
                data_root=layout.root,
                ingesta_sha256=sha256,
                declarada=fuente.datos_sinteticos,
                ingesta_ref_m1=informe.published.sha256,
            )
        except OrigenIncompatibleError as exc:
            error = exc
            informe = _informe_fallido(fuente, deposito, exc, layout.root)
    elif isinstance(error, RawConflictError):
        informe = _informe_conflicto(fuente, error, layout.root)
    else:
        assert error is not None
        informe = _informe_fallido(fuente, deposito, error, layout.root)
    ruta = write_ingest_report(informe, layout.root)
    return ReportedIngest(report=informe, path=ruta, result=resultado, error=error)
