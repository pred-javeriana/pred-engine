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
from typing import Literal, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from pred_engine.comun.exclusiones import SkuExclusion
from pred_engine.comun.llm.errores import LlmError, LlmProviderError, LlmTimeoutError
from pred_engine.comun.llm.protocolo import LlmProvider
from pred_engine.comun.modelos import (
    ADI_THRESHOLD,
    CV2_THRESHOLD,
    DiagnosticEntry,
    HeaderDiagnostic,
    TopologyMetrics,
)
from pred_engine.comun.reserva import ReserveCut
from pred_engine.ingesta.categorizacion import (
    TopologyArtifact,
    TopologyContractError,
    TopologyMathError,
    TopologyRoutingError,
)
from pred_engine.ingesta.continuidad.errores import TemporalContinuityError
from pred_engine.ingesta.data import (
    DepositResult,
    OnConflict,
    RawConflictError,
    RawWritePermissionError,
    default_data_root,
    deposit_raw,
    ensure_data_layout,
)
from pred_engine.ingesta.lector.hashing import hash_sha256_archivo
from pred_engine.ingesta.pipeline import (
    CsvIngestResult,
    IngestResult,
    ingest_canonical_csv,
    run_classify_parquet,
    run_ingest,
)
from pred_engine.ingesta.salida.errores import (
    HandoffPreconditionError,
    OutputContractError,
    PanelPreservationError,
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

    @classmethod
    def of(cls, corte: ReserveCut) -> ReserveInfo:
        return cls(
            t_star=corte.t_star.date().isoformat(),
            first_reserved=corte.first_reserved.date().isoformat(),
            last_observed=corte.last_observed.date().isoformat(),
            reserved_days=corte.reserved_days,
            fraction=corte.fraction,
        )

    def cut(self) -> ReserveCut:
        """Corte publicado por M1, validado al releer el informe."""
        corte = ReserveCut(
            t_star=cast(pd.Timestamp, pd.Timestamp(self.t_star)),
            first_reserved=cast(pd.Timestamp, pd.Timestamp(self.first_reserved)),
            last_observed=cast(pd.Timestamp, pd.Timestamp(self.last_observed)),
            reserved_days=self.reserved_days,
            fraction=self.fraction,
        )
        if (
            ReserveInfo.of(corte) != self
            or corte.first_reserved != corte.t_star + pd.Timedelta(days=1)
            or corte.reserved_dates[-1] != corte.last_observed
        ):
            raise ValueError("el corte del informe M1 no es coherente")
        return corte


class ThresholdsInfo(_Modelo):
    adi: float
    cv2: float


class TopologyInfo(_Modelo):
    thresholds: ThresholdsInfo
    metrics: list[TopologyMetrics]


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
    exclusions: list[SkuExclusion] = Field(default_factory=list)
    excluded_count: int = Field(default=0, ge=0)
    publications: list[PublishedInfo] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ReportedIngest:
    """Resultado de ``ingest_csv_with_report``: el informe y lo que lo origino."""

    report: IngestReportFile
    path: Path
    result: CsvIngestResult | IngestResult | None
    error: Exception | None


# Codigos de la seccion 11 del contrato, deducidos del tipo de la excepcion (G6 los
# formaliza; lo que no aparece aqui cae en ``pipeline.stage_failed``).
_CODIGOS: tuple[tuple[type[BaseException], str], ...] = (
    (RawConflictError, "ingest.raw_conflict"),
    (RawWritePermissionError, "ingest.raw_write_blocked"),
    (SemanticAlignmentError, "ingest.header_rejected"),
    (SchemaBarrierError, "ingest.schema_barrier"),
    (TemporalContinuityError, "ingest.temporal_continuity"),
    (TopologyMathError, "topology.math"),
    (TopologyRoutingError, "topology.routing"),
    (TopologyContractError, "topology.contract"),
    (OutputContractError, "handoff.contract"),
    (PanelPreservationError, "handoff.preservation"),
    (HandoffPreconditionError, "handoff.precondition"),
    (LlmError, "ingest.llm_provider"),
    (FileNotFoundError, "ingest.file_not_found"),
)


def error_code(exc: BaseException) -> str:
    """Codigo estable para ``exc`` (ver seccion 11 del contrato)."""
    for tipo, codigo in _CODIGOS:
        if isinstance(exc, tipo):
            return codigo
    if isinstance(exc, ValueError) and ".csv" in str(exc):
        return "ingest.not_csv"
    return "pipeline.stage_failed"


def report_path(data_root: str | Path | None, sha256: str) -> Path:
    """Ruta del informe de la ingesta cuyo CSV tiene ese SHA-256."""
    return ensure_data_layout(data_root).ingestas / f"{sha256}.json"


def son_informes_equivalentes(a: IngestReportFile, b: IngestReportFile) -> bool:
    """Comprueba si dos informes representan publicaciones equivalentes.

    Dos informes son equivalentes si ambos fueron aceptados y coinciden en:
    - Procedencia/origen (datos_sinteticos).
    - Corte publicado (reserve).
    - Exclusiones y conteo de exclusiones (exclusions, excluded_count).
    - Huella del Parquet publicado.
    """
    if a.status != "accepted" or b.status != "accepted":
        return False
    if a.source_file.datos_sinteticos != b.source_file.datos_sinteticos:
        return False
    if a.reserve != b.reserve:
        return False
    if a.exclusions != b.exclusions or a.excluded_count != b.excluded_count:
        return False
    pubs_a = {
        p.sha256
        for p in ([a.published] if a.published is not None else []) + a.publications
        if p is not None and p.sha256 is not None
    }
    pubs_b = {
        p.sha256
        for p in ([b.published] if b.published is not None else []) + b.publications
        if p is not None and p.sha256 is not None
    }
    if not (pubs_a & pubs_b):
        return False
    return True


def informe_publicado(
    data_root: str | Path | None, parquet_path: str | Path, huella: str
) -> IngestReportFile | None:
    """Informe M1 aceptado que publico esos bytes en esa misma ruta.

    No crea directorios. Unos bytes iguales en otra ruta no bastan: son otra
    publicacion. Si varias ingestas publicaron la misma ruta y huella, la
    procedencia es ambigua y se rechaza salvo que sean publicaciones equivalentes.
    """
    raiz = Path(data_root) if data_root is not None else default_data_root()
    destino = Path(parquet_path).expanduser().resolve()
    ligados: dict[str, IngestReportFile] = {}
    for ruta in (raiz / REPORTS_DIR).glob("[0-9a-f]" * 64 + ".json"):
        informe = IngestReportFile.model_validate_json(ruta.read_text())
        if informe.status != "accepted":
            continue
        publicaciones = (
            [informe.published] if informe.published is not None else []
        ) + informe.publications
        coincide = False
        for publicado in publicaciones:
            if publicado is None or publicado.sha256 != huella:
                continue
            origen = Path(publicado.parquet_path)
            if (origen if origen.is_absolute() else raiz / origen).resolve() == destino:
                coincide = True
                break
        if coincide:
            ligados[informe.ingest_id] = informe
    if len(ligados) > 1:
        informes = list(ligados.values())
        if all(son_informes_equivalentes(informes[0], otro) for otro in informes[1:]):
            return min(informes, key=lambda r: r.ingest_id)
        raise ValueError(
            f"el Parquet {destino} tiene varias procedencias M1: "
            + ", ".join(sorted(ligados))
        )
    return next(iter(ligados.values()), None)


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
    corte = artefacto.reserve
    reserva = ReserveInfo.of(
        corte if corte is not None else ReserveCut.of(artefacto.frame)
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
    deposito: DepositResult | None,
    topologia: TopologyArtifact,
    parquet: Path,
    raiz: Path,
    *,
    validadas: int,
    cabecera: HeaderInfo | None = None,
) -> IngestReportFile:
    reserva, info = _topologia(topologia)
    n_skus = len(topologia.metrics)
    publicado = PublishedInfo(
        sha256=hash_sha256_archivo(parquet),
        parquet_path=_relativa(parquet, raiz),
        rows=len(topologia.frame),
        n_skus=n_skus,
        columns=[str(c) for c in topologia.frame.columns],
    )
    return IngestReportFile(
        written_at=_ahora(),
        ingest_id=fuente.sha256,
        status="accepted",
        source_file=fuente,
        deposit=None if deposito is None else _deposito_info(deposito, raiz),
        header_diagnostic=cabecera,
        validation=ValidationInfo(
            rows_validated=validadas,
            rows_daily_panel=len(topologia.frame),
            n_skus=n_skus,
        ),
        published=publicado,
        reserve=reserva,
        topology=info,
        exclusions=list(topologia.exclusions),
        excluded_count=len(topologia.exclusions),
        publications=[publicado],
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
    from pred_engine.comun.procedencia import resolver_origen

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
    if destino.exists():
        try:
            previa = IngestReportFile.model_validate_json(destino.read_text())
        except Exception:
            previa = None
        if previa is not None and previa.status == "accepted":
            viejas = (
                [previa.published] if previa.published is not None else []
            ) + previa.publications
            nuevas = (
                [report.published] if report.published is not None else []
            ) + report.publications
            todas: dict[str, PublishedInfo] = {}
            for p in viejas + nuevas:
                if p is not None:
                    todas[p.parquet_path] = p
            report = report.model_copy(update={"publications": list(todas.values())})
    elif report.published is not None and not report.publications:
        report = report.model_copy(update={"publications": [report.published]})
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
            resultado = ingest_canonical_csv(
                deposito.path,
                data_root=layout.root,
                datos_sinteticos=datos_sinteticos,
                deposito=deposito,
            )
        else:
            resultado = run_ingest(
                deposito.path,
                provider,
                data_root=layout.root,
                timeout=timeout,
                datos_sinteticos=datos_sinteticos,
                deposito=deposito,
            )
    except Exception as exc:  # noqa: BLE001 - se registra en el informe y se devuelve
        error = exc

    if resultado is not None:
        ruta = report_path(layout.root, sha256)
        informe = IngestReportFile.model_validate_json(ruta.read_text())
    elif isinstance(error, RawConflictError):
        informe = _informe_conflicto(fuente, error, layout.root)
        ruta = write_ingest_report(informe, layout.root)
    else:
        assert error is not None
        informe = _informe_fallido(fuente, deposito, error, layout.root)
        ruta = write_ingest_report(informe, layout.root)
    return ReportedIngest(report=informe, path=ruta, result=resultado, error=error)


def classify_parquet_with_report(
    parquet_path: str | Path,
    *,
    data_root: str | Path | None = None,
    datos_sinteticos: bool | None = None,
) -> tuple[TopologyArtifact, Path, Path]:
    """``run_classify_parquet`` que deja su informe M1 junto al Parquet publicado.

    El informe usa el mismo esquema que la ruta CSV, sin deposito en raw/, y
    lleva las exclusiones para que ``run --parquet`` las recupere. Los errores
    se propagan sin informe, como en ``run_classify_parquet``. Devuelve la
    topologia, el Parquet publicado y la ruta del informe.
    """
    layout = ensure_data_layout(data_root)
    origen = Path(parquet_path).expanduser().resolve()
    if not origen.is_file():
        raise FileNotFoundError(origen)
    sha256 = hash_sha256_archivo(origen)
    # Validar origen ANTES de run_classify_parquet (S4)
    origen_resuelto = origen_informe(layout.root, sha256, datos_sinteticos)
    topologia, destino = run_classify_parquet(
        origen, data_root=layout.root, datos_sinteticos=origen_resuelto
    )
    ruta = report_path(layout.root, sha256)
    return topologia, destino, ruta
