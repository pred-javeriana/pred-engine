"""Resolucion de rutas y hashing para la Fase 0.

Este modulo es deliberadamente autonomo: la Fase 0 opera bajo una frontera
arquitectonica estricta y no importa componentes del Framework PRED (Modulo 1),
por lo que replica aqui la resolucion de la raiz de datos en lugar de reutilizar
``pred_engine.ingesta``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

_ENV_RAIZ = "PRED_DATA_ROOT"
_TAMANO_BLOQUE = 65536


@dataclass(frozen=True, slots=True)
class RutasFase0:
    """Rutas derivadas de la raiz de datos inyectada."""

    root: Path
    raw: Path
    logs: Path


def resolver_raiz_datos(data_root: str | Path | None = None) -> Path:
    """Resuelve la raiz de datos sin asumir rutas absolutas del sistema."""
    if data_root is not None:
        return Path(data_root)
    configurada = os.environ.get(_ENV_RAIZ)
    return Path(configurada) if configurada else Path("data")


def resolver_rutas(data_root: str | Path | None = None) -> RutasFase0:
    """Devuelve ``raw/`` (WORM) y ``logs/`` creando el arbol si hace falta."""
    raiz = resolver_raiz_datos(data_root)
    rutas = RutasFase0(root=raiz, raw=raiz / "raw", logs=raiz / "logs")
    rutas.raw.mkdir(parents=True, exist_ok=True)
    rutas.logs.mkdir(parents=True, exist_ok=True)
    return rutas


def hash_sha256_archivo(
    ruta: str | Path,
    tamano_bloque: int = _TAMANO_BLOQUE,
) -> str:
    """Calcula SHA-256 del archivo en disco sin cargarlo completo en RAM."""
    digest = hashlib.sha256()
    with Path(ruta).open("rb") as fh:
        while bloque := fh.read(tamano_bloque):
            digest.update(bloque)
    return digest.hexdigest()
