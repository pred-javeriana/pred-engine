"""Deposito del CSV original en raw/ sin pisar archivos ya depositados (G5).

El deposito es la unica escritura permitida en raw/ y es de solo adicion
(ADR-01-002): un archivo presente jamas se modifica ni se borra. Si el nombre ya
existe con otro contenido, el llamador decide: rechazar (``reject``) o conservar
ambos (``keep_both``), con el nuevo bajo un nombre derivado de su SHA-256.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import Literal

from pred_engine.comun.logger import get_logger
from pred_engine.ingesta.data.layout import ensure_data_layout

_logger = get_logger(__name__)

DepositStatus = Literal[
    "deposited", "already_present_identical", "already_present_different"
]
OnConflict = Literal["reject", "keep_both"]

_LONGITUD_SUFIJO = 8


class RawConflictError(FileExistsError):
    """Ya existe en raw/ un archivo con ese nombre y contenido distinto.

    No se escribio nada. Con ``on_conflict="keep_both"`` el archivo nuevo se
    deposita con un nombre versionado y el original queda intacto.
    """

    def __init__(
        self, existing: Path, existing_sha256: str, incoming_sha256: str
    ) -> None:
        self.existing = existing
        self.existing_sha256 = existing_sha256
        self.incoming_sha256 = incoming_sha256
        super().__init__(
            f"raw/{existing.name} ya existe con otro contenido "
            f"(existente {existing_sha256[:12]}..., nuevo {incoming_sha256[:12]}...); "
            "confirme para conservar ambos"
        )


@dataclass(frozen=True, slots=True)
class DepositResult:
    """Resultado de depositar un archivo.

    ``would_overwrite`` indica una colision de nombre con contenido distinto: el
    archivo no se sobrescribe nunca; con ``keep_both`` queda bajo otro nombre.
    """

    path: Path
    sha256: str
    status: DepositStatus
    would_overwrite: bool


def versioned_name(name: str, sha256: str) -> str:
    """``datos.csv`` + hash -> ``datos-<8 primeros del hash>.csv``."""
    ruta = Path(name)
    return f"{ruta.stem}-{sha256[:_LONGITUD_SUFIJO]}{ruta.suffix}"


def _copiar_atomico(origen: Path, destino: Path) -> None:
    """Copia a un temporal del mismo directorio y renombra; no deja parciales."""
    temporal = destino.with_name(f".{destino.name}.{os.getpid()}.part")
    try:
        copy2(origen, temporal)
        os.replace(temporal, destino)
    finally:
        temporal.unlink(missing_ok=True)


def deposit_raw(
    source: str | Path,
    *,
    data_root: str | Path | None = None,
    on_conflict: OnConflict = "reject",
) -> DepositResult:
    """Deposita ``source`` en raw/ y devuelve donde quedo y por que.

    Raises:
        FileNotFoundError: ``source`` no existe.
        RawConflictError: el nombre existe con otro contenido y ``on_conflict``
            es ``"reject"``. No se escribe nada.
    """
    # Import diferido: `lector` depende de este paquete y cerraria un ciclo.
    from pred_engine.ingesta.lector.hashing import hash_sha256_archivo

    layout = ensure_data_layout(data_root)
    origen = Path(source).expanduser().resolve()
    if not origen.is_file():
        raise FileNotFoundError(origen)
    sha = hash_sha256_archivo(origen)
    destino = (layout.raw / origen.name).resolve()

    if origen == destino:
        return DepositResult(destino, sha, "already_present_identical", False)

    if not destino.exists():
        _copiar_atomico(origen, destino)
        _logger.info("Archivo depositado en raw: %s sha256=%s", destino.name, sha)
        return DepositResult(destino, sha, "deposited", False)

    existente = hash_sha256_archivo(destino)
    if existente == sha:
        return DepositResult(destino, sha, "already_present_identical", False)

    if on_conflict == "reject":
        raise RawConflictError(destino, existente, sha)

    versionado = (layout.raw / versioned_name(origen.name, sha)).resolve()
    if not versionado.exists():
        _copiar_atomico(origen, versionado)
        _logger.info(
            "Conflicto de nombre en raw: se conservan ambos (%s y %s)",
            destino.name,
            versionado.name,
        )
        return DepositResult(versionado, sha, "deposited", True)
    if hash_sha256_archivo(versionado) != sha:
        # Colision de 8 caracteres de hash: no se pisa nada.
        raise RawConflictError(versionado, hash_sha256_archivo(versionado), sha)
    return DepositResult(versionado, sha, "already_present_identical", True)
