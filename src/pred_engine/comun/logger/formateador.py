"""Formateador JSON para el registrador estructurado de PRED."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Serializa cada registro como un unico objeto JSON por linea."""

    def format(self, record: logging.LogRecord) -> str:
        # Marca de tiempo ISO-8601 en UTC, derivada del evento original.
        marca = datetime.fromtimestamp(record.created, tz=UTC).isoformat()
        carga: dict[str, Any] = {
            "timestamp": marca,
            "level": record.levelname,
            "module": record.module,
            "file_hash": getattr(record, "file_hash", None),
            "row_count": getattr(record, "row_count", None),
            "message": record.getMessage(),
        }
        if record.exc_info:
            # El traceback es diagnostico; no sustituye los campos obligatorios.
            carga["exception"] = self.formatException(record.exc_info)
        return json.dumps(carga, ensure_ascii=False, default=str)
