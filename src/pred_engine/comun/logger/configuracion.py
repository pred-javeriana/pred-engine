"""Configuracion del registrador JSON hacia stdout."""

from __future__ import annotations

import logging
import sys
from typing import TextIO

from pred_engine.comun.logger.formateador import JsonFormatter

_LOGGER_APLICACION = "pred_engine"


def configure_json_logger(
    name: str = _LOGGER_APLICACION,
    *,
    level: int = logging.INFO,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Devuelve un logger que emite JSON en stdout (o en `stream` de prueba)."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Evita duplicar lineas JSON en el logger raiz del proceso anfitrion.
    logger.propagate = False

    destino = sys.stdout if stream is None else stream
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and isinstance(
            handler.formatter, JsonFormatter
        ):
            return logger

    handler = logging.StreamHandler(destino)
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Obtiene un logger hijo bajo el namespace JSON de PRED."""
    configure_json_logger(_LOGGER_APLICACION)
    return logging.getLogger(name or _LOGGER_APLICACION)


def log_ingestion_event(
    logger: logging.Logger,
    message: str,
    *,
    file_hash: str,
    row_count: int,
    level: int = logging.INFO,
) -> None:
    """Emite un evento de ingesta con hash y filas en la carga JSON."""
    logger.log(
        level,
        message,
        extra={"file_hash": file_hash, "row_count": row_count},
    )
