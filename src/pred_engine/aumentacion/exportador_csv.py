"""0.4-B2 - Exportador CSV del artefacto final de la Fase 0.

Materializa el panel ya aprobado por la compuerta de restricciones y por el
validador de esquema en un unico CSV, delegando la escritura fisica en la
guarda WORM del directorio crudo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pred_engine.aumentacion.conformidad import validar_conformidad_o_fallar
from pred_engine.aumentacion.contrato import OUTPUT_COLUMNS, TIMESTAMP_FORMAT
from pred_engine.aumentacion.rutas import hash_sha256_archivo
from pred_engine.aumentacion.worm import escribir_una_sola_vez
from pred_engine.comun.logger import get_logger, log_ingestion_event

_logger = get_logger(__name__)

# Umbral de la seccion 0.4: la Fase 0 entrega un entorno de estres a escala
# comercial (50 000+ registros).
MINIMO_FILAS_POR_DEFECTO = 50_000
NOMBRE_ARTEFACTO_POR_DEFECTO = "panel_sintetico_fase0.csv"


@dataclass(frozen=True, slots=True)
class ArtefactoExportado:
    """Metadatos del CSV depositado en el directorio crudo."""

    path: Path
    sha256: str
    row_count: int


def _preparar_marco(frame: pd.DataFrame) -> pd.DataFrame:
    """Reordena al orden canonico y formatea la marca de tiempo a ISO 8601."""
    ordenado = frame.loc[:, list(OUTPUT_COLUMNS)].copy()
    ordenado["timestamp"] = pd.to_datetime(ordenado["timestamp"]).dt.strftime(
        TIMESTAMP_FORMAT
    )
    return ordenado


def exportar_artefacto_csv(
    frame: pd.DataFrame,
    nombre: str = NOMBRE_ARTEFACTO_POR_DEFECTO,
    *,
    data_root: str | Path | None = None,
    minimo_filas: int = MINIMO_FILAS_POR_DEFECTO,
) -> ArtefactoExportado:
    """Valida, serializa y deposita el artefacto CSV bajo politica WORM."""
    reporte = validar_conformidad_o_fallar(frame)

    if minimo_filas and reporte.row_count < minimo_filas:
        raise ValueError(
            f"el artefacto tiene {reporte.row_count} filas; "
            f"se exigen al menos {minimo_filas}"
        )

    preparado = _preparar_marco(frame)

    def _escribir(destino: Path) -> None:
        preparado.to_csv(destino, index=False, encoding="utf-8")

    destino = escribir_una_sola_vez(nombre, _escribir, data_root=data_root)
    digest = hash_sha256_archivo(destino)

    log_ingestion_event(
        _logger,
        "Artefacto de la Fase 0 exportado",
        file_hash=digest,
        row_count=reporte.row_count,
    )
    return ArtefactoExportado(
        path=destino,
        sha256=digest,
        row_count=reporte.row_count,
    )
