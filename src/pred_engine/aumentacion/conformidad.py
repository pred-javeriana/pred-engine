"""0.4-A2 - Validador de conformidad de esquema.

Verifica el DataFrame consolidado contra el contrato de datos antes de su
escritura. Ante cualquier desviacion detiene la exportacion con una excepcion
critica. El reporte de conformidad se genera en toda ejecucion, exitosa o no.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api import types as pdt

from pred_engine.aumentacion.contrato import (
    CONTRACT_VERSION,
    OUTPUT_COLUMNS,
    OUTPUT_CONTRACT,
)
from pred_engine.aumentacion.errores import SchemaConformanceError
from pred_engine.comun.logger import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Verificacion:
    """Resultado de una comprobacion individual del contrato."""

    nombre: str
    ok: bool
    detalle: str


@dataclass(frozen=True, slots=True)
class ReporteConformidad:
    """Reporte agregado de conformidad de esquema."""

    conforme: bool
    contract_version: str
    row_count: int
    verificaciones: tuple[Verificacion, ...]

    @property
    def fallas(self) -> tuple[str, ...]:
        return tuple(
            f"{v.nombre}: {v.detalle}" for v in self.verificaciones if not v.ok
        )


def _dtype_conforme(serie: pd.Series, esperado: str) -> bool:
    if esperado.startswith("datetime64"):
        return pdt.is_datetime64_any_dtype(serie)
    if esperado == "int64":
        return pdt.is_integer_dtype(serie)
    if esperado == "string":
        return pdt.is_string_dtype(serie) or pdt.is_object_dtype(serie)
    return str(serie.dtype) == esperado


def verificar_conformidad(frame: pd.DataFrame) -> ReporteConformidad:
    """Construye el reporte de conformidad sin elevar excepciones."""
    verificaciones: list[Verificacion] = []
    columnas = list(frame.columns)

    faltantes = [c for c in OUTPUT_COLUMNS if c not in columnas]
    sobrantes = [c for c in columnas if c not in OUTPUT_COLUMNS]
    verificaciones.append(
        Verificacion(
            nombre="columnas_presentes",
            ok=not faltantes and not sobrantes,
            detalle=(
                "conjunto exacto de columnas"
                if not faltantes and not sobrantes
                else f"faltantes={faltantes} sobrantes={sobrantes}"
            ),
        )
    )
    verificaciones.append(
        Verificacion(
            nombre="orden_canonico",
            ok=columnas == list(OUTPUT_COLUMNS),
            detalle=(
                "orden correcto"
                if columnas == list(OUTPUT_COLUMNS)
                else f"esperado={list(OUTPUT_COLUMNS)} recibido={columnas}"
            ),
        )
    )

    for col in OUTPUT_CONTRACT:
        if col.name not in columnas:
            verificaciones.append(
                Verificacion(
                    nombre=f"tipo::{col.name}",
                    ok=False,
                    detalle="columna ausente",
                )
            )
            continue
        ok = _dtype_conforme(frame[col.name], col.dtype)
        verificaciones.append(
            Verificacion(
                nombre=f"tipo::{col.name}",
                ok=ok,
                detalle=(
                    f"{col.dtype}"
                    if ok
                    else f"esperado={col.dtype} recibido={frame[col.name].dtype}"
                ),
            )
        )

    if {"demand_qty", "lead_time_days"}.issubset(columnas):
        demanda_ok = bool((frame["demand_qty"] >= 0).all())
        lead_ok = bool((frame["lead_time_days"] >= 1).all())
        verificaciones.append(
            Verificacion(
                nombre="ley_demanda_no_negativa",
                ok=demanda_ok,
                detalle="demand_qty >= 0" if demanda_ok else "hay demanda negativa",
            )
        )
        verificaciones.append(
            Verificacion(
                nombre="ley_lead_time_minimo",
                ok=lead_ok,
                detalle="lead_time_days >= 1" if lead_ok else "hay lead time < 1",
            )
        )

    conforme = all(v.ok for v in verificaciones)
    reporte = ReporteConformidad(
        conforme=conforme,
        contract_version=CONTRACT_VERSION,
        row_count=int(len(frame)),
        verificaciones=tuple(verificaciones),
    )
    _logger.info(
        "Reporte de conformidad: %s (%s filas, %s verificaciones)",
        "CONFORME" if conforme else "NO CONFORME",
        reporte.row_count,
        len(verificaciones),
    )
    return reporte


def validar_conformidad_o_fallar(frame: pd.DataFrame) -> ReporteConformidad:
    """Genera el reporte y eleva :class:`SchemaConformanceError` si no conforma."""
    reporte = verificar_conformidad(frame)
    if not reporte.conforme:
        _logger.critical(
            "Artefacto no conforme al contrato v%s: %s",
            reporte.contract_version,
            "; ".join(reporte.fallas),
        )
        raise SchemaConformanceError(
            "el artefacto consolidado no cumple el contrato de datos de la Fase 0",
            fallas=reporte.fallas,
        )
    return reporte
