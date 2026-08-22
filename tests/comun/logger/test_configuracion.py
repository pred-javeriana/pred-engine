"""Pruebas de configuracion del logger JSON y telemetria de ingesta."""

from __future__ import annotations

import io
import json
import logging
import sys

from pred_engine.comun.logger import (
    JsonFormatter,
    configure_json_logger,
    log_ingestion_event,
)


def test_handler_escribe_en_stdout_por_defecto() -> None:
    logger = configure_json_logger("pred_engine.test.stdout")
    assert any(
        isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout
        for handler in logger.handlers
    )


def test_emite_json_valido_en_el_stream() -> None:
    buffer = io.StringIO()
    logger = configure_json_logger("pred_engine.test.buffer", stream=buffer)
    logger.info("hola")
    carga = json.loads(buffer.getvalue())
    assert carga["message"] == "hola"
    assert carga["level"] == "INFO"


def test_evento_de_ingesta_incluye_hash_y_filas() -> None:
    buffer = io.StringIO()
    logger = configure_json_logger("pred_engine.test.ingest", stream=buffer)
    log_ingestion_event(
        logger,
        "extraccion completada",
        file_hash="deadbeef",
        row_count=12,
    )
    carga = json.loads(buffer.getvalue())
    assert carga["file_hash"] == "deadbeef"
    assert carga["row_count"] == 12


def test_configuracion_idempotente_no_duplica_handlers() -> None:
    nombre = "pred_engine.test.idempotente"
    primero = configure_json_logger(nombre)
    segundo = configure_json_logger(nombre)
    assert primero is segundo
    json_handlers = [
        handler
        for handler in primero.handlers
        if isinstance(handler.formatter, JsonFormatter)
    ]
    assert len(json_handlers) == 1
