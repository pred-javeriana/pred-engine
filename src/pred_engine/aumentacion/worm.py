"""0.4-B1 - Guarda de escritura WORM (Write-Once-Read-Many).

Impide la sobrescritura de artefactos ya depositados en el directorio crudo.
Toda ruta se deriva de forma relativa a la raiz de datos inyectada por
variable de entorno; no se admiten rutas absolutas ni nombres con separadores.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PurePosixPath, PureWindowsPath

from pred_engine.aumentacion.errores import WormOverwriteError
from pred_engine.aumentacion.rutas import resolver_rutas
from pred_engine.comun.logger import get_logger

_logger = get_logger(__name__)

EscritorArtefacto = Callable[[Path], None]


def _validar_nombre(nombre: str) -> str:
    """Rechaza rutas absolutas y nombres con componentes de directorio."""
    if not nombre or not nombre.strip():
        raise ValueError("el nombre del artefacto no puede estar vacio")
    candidato = nombre.strip()
    puro_posix = PurePosixPath(candidato)
    puro_win = PureWindowsPath(candidato)
    if puro_posix.is_absolute() or puro_win.is_absolute():
        raise ValueError(
            f"el nombre del artefacto no puede ser una ruta absoluta: {nombre}"
        )
    if len(puro_posix.parts) != 1 or len(puro_win.parts) != 1:
        raise ValueError(
            f"el nombre del artefacto no puede contener directorios: {nombre}"
        )
    if candidato in {".", ".."}:
        raise ValueError(f"nombre de artefacto invalido: {nombre}")
    return candidato


def resolver_ruta_artefacto(
    nombre: str,
    *,
    data_root: str | Path | None = None,
) -> Path:
    """Devuelve la ruta destino dentro de ``{data_root}/raw/`` para ``nombre``."""
    rutas = resolver_rutas(data_root)
    return rutas.raw / _validar_nombre(nombre)


def escribir_una_sola_vez(
    nombre: str,
    escritor: EscritorArtefacto,
    *,
    data_root: str | Path | None = None,
) -> Path:
    """Ejecuta ``escritor(destino)`` solo si el artefacto no existe todavia.

    Eleva :class:`WormOverwriteError` cuando el archivo destino ya existe,
    preservando la inmutabilidad del almacenamiento crudo.
    """
    destino = resolver_ruta_artefacto(nombre, data_root=data_root)
    if destino.exists():
        _logger.critical("Violacion WORM: intento de sobrescritura en %s", destino)
        raise WormOverwriteError(destino)

    escritor(destino)

    if not destino.is_file():
        raise RuntimeError(
            f"el escritor no materializo el artefacto esperado en {destino}"
        )
    _logger.info("Artefacto depositado bajo politica WORM: %s", destino)
    return destino
