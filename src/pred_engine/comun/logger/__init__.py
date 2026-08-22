"""Registrador JSON estructurado compartido por todos los modulos PRED."""

from pred_engine.comun.logger.configuracion import (
    configure_json_logger,
    get_logger,
    log_ingestion_event,
)
from pred_engine.comun.logger.formateador import JsonFormatter

__all__ = [
    "JsonFormatter",
    "configure_json_logger",
    "get_logger",
    "log_ingestion_event",
]
