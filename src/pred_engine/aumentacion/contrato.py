"""0.4-A1 - Contrato de datos del artefacto de salida de la Fase 0.

El CSV que la Fase 0 deposita en ``data/raw/`` simula el estandar de
exportacion de un ERP. El Modulo 1 (ingesta agnostica) lo consume sin conocer
su origen simulado, por lo que el esquema es estricto y versionado.
"""

from __future__ import annotations

from dataclasses import dataclass

# Version del contrato. Incrementar ante cualquier cambio de columnas o tipos.
CONTRACT_VERSION = "1.0.0"

# Formato canonico de la marca de tiempo (ISO 8601, fecha sin hora).
TIMESTAMP_FORMAT = "%Y-%m-%d"


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """Especificacion de una columna del artefacto de salida."""

    name: str
    dtype: str
    description: str


# Orden canonico de columnas del CSV exportado. Sin campos adicionales.
OUTPUT_CONTRACT: tuple[ColumnSpec, ...] = (
    ColumnSpec(
        name="sku_id",
        dtype="string",
        description="Identificador unico del producto medico o insumo logistico.",
    ),
    ColumnSpec(
        name="timestamp",
        dtype="datetime64[ns]",
        description="Fecha del consolidado diario en formato ISO 8601 (YYYY-MM-DD).",
    ),
    ColumnSpec(
        name="demand_qty",
        dtype="int64",
        description="Cantidad fisica demandada. Garantizada como entero >= 0.",
    ),
    ColumnSpec(
        name="lead_time_days",
        dtype="int64",
        description="Tiempo de entrega historico en dias. Entero >= 1.",
    ),
)

# Nombres en orden canonico, para reordenar y validar DataFrames.
OUTPUT_COLUMNS: tuple[str, ...] = tuple(col.name for col in OUTPUT_CONTRACT)

# Indice por nombre para consultas puntuales.
CONTRACT_BY_NAME: dict[str, ColumnSpec] = {col.name: col for col in OUTPUT_CONTRACT}


def describir_contrato() -> str:
    """Render legible del contrato para documentacion y bitacoras."""
    lineas = [f"Contrato de datos de la Fase 0 (v{CONTRACT_VERSION})"]
    for indice, col in enumerate(OUTPUT_CONTRACT, start=1):
        lineas.append(f"  {indice}. {col.name} ({col.dtype}) - {col.description}")
    return "\n".join(lineas)
