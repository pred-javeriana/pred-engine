"""Aplicacion vectorizada del diccionario de mapeo y sanitizacion de columnas."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos import CANONICAL_FIELDS, HeaderMapping
from pred_engine.ingesta.sonda.errores import SemanticAlignmentError

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AlignmentArtifact:
    """Marco alineado al contrato mas el mapeo auditado."""

    frame: pd.DataFrame
    mapping: HeaderMapping
    dropped_columns: tuple[str, ...]


def _resolver_nombre_fuente(solicitado: str, columnas: list[str]) -> str:
    if solicitado in columnas:
        return solicitado
    bajos = {c.lower(): c for c in columnas}
    if (
        solicitado.lower() in bajos
        and list(c.lower() for c in columnas).count(solicitado.lower()) == 1
    ):
        return bajos[solicitado.lower()]
    raise SemanticAlignmentError(
        "PRED no puede trabajar con este dataset: el modelo de lenguaje "
        f"propuso la columna {solicitado!r}, que no existe en el archivo."
    )


def apply_header_mapping(
    frame: pd.DataFrame,
    mapping: HeaderMapping,
) -> AlignmentArtifact:
    """rename vectorizado + drop de no mapeadas. Fail-closed si falta un campo."""
    faltantes = mapping.unmapped_fields()
    if faltantes:
        _logger.error("Mapeo incompleto, campos=%s", list(faltantes))
        raise SemanticAlignmentError(
            "PRED no puede trabajar con este dataset: no se identificaron "
            "con confianza las columnas "
            + ", ".join(faltantes)
            + ". No se inventara un mapeo."
        )
    columnas = [str(c) for c in frame.columns]
    rename: dict[str, str] = {}
    for canonico in CANONICAL_FIELDS:
        fuente = getattr(mapping, canonico)
        real = _resolver_nombre_fuente(fuente, columnas)
        if real in rename and rename[real] != canonico:
            raise SemanticAlignmentError(
                "PRED no puede trabajar con este dataset: "
                f"{real!r} mapea a dos campos canonico."
            )
        rename[real] = canonico
    # df.rename es vectorizado; no se itera fila a fila.
    alineado = frame.rename(columns=rename)
    dropped = tuple(c for c in alineado.columns if c not in CANONICAL_FIELDS)
    sanitizado = alineado.loc[:, list(CANONICAL_FIELDS)].copy()
    _logger.info(
        "Alineacion semantica ok; descartadas=%s",
        list(dropped),
    )
    return AlignmentArtifact(
        frame=sanitizado,
        mapping=mapping,
        dropped_columns=dropped,
    )
