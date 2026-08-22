"""Pruebas del formateador JSON estructurado."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from pred_engine.comun.logger.formateador import JsonFormatter


def _registro(
    mensaje: str = "ingesta ok",
    nivel: int = logging.INFO,
) -> logging.LogRecord:
    return logging.LogRecord(
        name="pred_engine",
        level=nivel,
        pathname="extractor_csv.py",
        lineno=1,
        msg=mensaje,
        args=(),
        exc_info=None,
    )


def test_salida_es_json_estrictamente_valido() -> None:
    texto = JsonFormatter().format(_registro())
    carga = json.loads(texto)
    assert isinstance(carga, dict)


def test_timestamp_iso8601_y_nivel() -> None:
    carga = json.loads(JsonFormatter().format(_registro()))
    datetime.fromisoformat(carga["timestamp"])
    assert carga["level"] == "INFO"


def test_campos_obligatorios_presentes_con_nulos() -> None:
    carga = json.loads(JsonFormatter().format(_registro()))
    assert "module" in carga
    assert carga["file_hash"] is None
    assert carga["row_count"] is None
    assert carga["message"] == "ingesta ok"


def test_telemetria_desde_extra() -> None:
    registro = _registro()
    registro.file_hash = "abc123"
    registro.row_count = 42
    carga = json.loads(JsonFormatter().format(registro))
    assert carga["file_hash"] == "abc123"
    assert carga["row_count"] == 42


def test_una_sola_linea_sin_pretty_print() -> None:
    texto = JsonFormatter().format(_registro())
    assert "\n" not in texto
