"""Publicacion del artefacto Parquet 1.4 hacia processed/."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from pred_engine.comun.exclusiones import SkuExclusion
from pred_engine.comun.logger import get_logger
from pred_engine.comun.metadatos_parquet import (
    ParquetMetadata,
    empaquetar_metadatos_schema,
    leer_metadatos_parquet,
)
from pred_engine.comun.reserva import ReserveCut
from pred_engine.ingesta.lector import export_parquet
from pred_engine.ingesta.salida.errores import OutputContractError
from pred_engine.ingesta.salida.preservacion import require_panel_preserved
from pred_engine.ingesta.salida.validador import (
    contar_skus,
    normalize_output_frame,
    validate_output_contract,
)

_logger = get_logger(__name__)


def publish_classified_panel(
    classified: pd.DataFrame,
    destination: str | Path,
    *,
    data_root: str | Path | None = None,
    daily_panel: pd.DataFrame,
    reserve: ReserveCut | None = None,
    exclusions: Sequence[SkuExclusion] = (),
) -> Path:
    """Valida preservacion + contrato y escribe Parquet en processed/.

    No importa ADI/CV² ni el enrutador SBC. No escribe CSV ni toca raw/.
    """
    destino = Path(destination)
    if destino.suffix.lower() != ".parquet":
        raise OutputContractError("el artefacto final solo se publica como .parquet")

    require_panel_preserved(daily_panel, classified)
    validate_output_contract(classified)
    normalizado = normalize_output_frame(classified)
    if reserve is None:
        try:
            reserve = ReserveCut.of(daily_panel)
        except Exception:
            reserve = None
    metadata = (
        empaquetar_metadatos_schema(reserve, exclusions)
        if reserve is not None
        else None
    )
    escrito = export_parquet(
        normalizado, destino, data_root=data_root, metadata=metadata
    )
    _logger.info(
        "Artefacto 1.4 publicado ruta=%s filas=%s skus=%s",
        escrito,
        int(len(normalizado)),
        contar_skus(cast(pd.Series, normalizado["sku_id"])),
    )
    return escrito


@dataclass(frozen=True, slots=True, eq=False)
class ClassifiedPublication:
    """Filas, SHA-256 y metadatos de corte obtenidos de una sola lectura del archivo.

    Una publicacion sustituida en la misma ruta no puede mezclar sus filas con
    el corte o las exclusiones de otra: todo sale de los mismos bytes.
    """

    frame: pd.DataFrame
    sha256: str
    metadata: ParquetMetadata


def read_classified_publication(path: str | Path) -> ClassifiedPublication:
    """Lee un Parquet una vez, impone el contrato 1.4 y exige sus metadatos.

    Un artefacto sin los metadatos de corte de pred_engine no es un handoff
    valido: se rechaza pidiendo reimportar desde el CSV. No escribe.
    """
    origen = Path(path).expanduser().resolve()
    if not origen.is_file():
        raise FileNotFoundError(origen)
    if origen.suffix.lower() != ".parquet":
        raise OutputContractError("solo se verifica un artefacto .parquet")
    datos = origen.read_bytes()
    marco = pd.read_parquet(io.BytesIO(datos), engine="pyarrow")
    validate_output_contract(marco)
    metadata = leer_metadatos_parquet(datos)
    _logger.info(
        "Artefacto 1.4 verificado ruta=%s filas=%s skus=%s",
        origen,
        int(len(marco)),
        contar_skus(cast(pd.Series, marco["sku_id"])),
    )
    return ClassifiedPublication(marco, hashlib.sha256(datos).hexdigest(), metadata)


def read_classified_parquet(path: str | Path) -> pd.DataFrame:
    """Lee un Parquet y vuelve a imponer el contrato 1.4. No escribe."""
    return read_classified_publication(path).frame


def read_classified_parquet_with_sha256(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Como ``read_classified_parquet``, con el SHA-256 de los mismos bytes leidos."""
    publicacion = read_classified_publication(path)
    return publicacion.frame, publicacion.sha256
