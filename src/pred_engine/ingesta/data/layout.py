"""Arbol de almacenamiento inmutable para la ingesta pasiva."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_ENV_RAIZ = "PRED_DATA_ROOT"


@dataclass(frozen=True, slots=True)
class DataLayout:
    """Rutas del contrato de directorios de ingesta."""

    root: Path
    raw: Path
    staging: Path
    processed: Path


def default_data_root() -> Path:
    """Resuelve la raiz de datos sin hardcodear rutas absolutas del SO."""
    configurada = os.environ.get(_ENV_RAIZ)
    return Path(configurada) if configurada else Path("data")


def ensure_data_layout(data_root: str | Path | None = None) -> DataLayout:
    """Crea raw/staging/processed si no existen y devuelve el layout."""
    raiz = Path(data_root) if data_root is not None else default_data_root()
    layout = DataLayout(
        root=raiz,
        raw=raiz / "raw",
        staging=raiz / "staging",
        processed=raiz / "processed",
    )
    for directorio in (layout.raw, layout.staging, layout.processed):
        # exist_ok garantiza idempotencia: no falla si el arbol ya existe.
        directorio.mkdir(parents=True, exist_ok=True)
    return layout
