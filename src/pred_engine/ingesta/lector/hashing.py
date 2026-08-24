"""Hash criptografico de artefactos crudos (funcion pura)."""

from __future__ import annotations

import hashlib
from pathlib import Path

_TAMANO_BLOQUE = 65536


def hash_sha256_archivo(ruta: str | Path, tamano_bloque: int = _TAMANO_BLOQUE) -> str:
    """Calcula SHA-256 del archivo en disco sin cargarlo completo en RAM."""
    digest = hashlib.sha256()
    with Path(ruta).open("rb") as fh:
        while bloque := fh.read(tamano_bloque):
            digest.update(bloque)
    return digest.hexdigest()
