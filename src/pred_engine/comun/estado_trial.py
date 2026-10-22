"""Estado visible del ensayo, independiente del estado usado por el backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pred_engine.comun.dataclasses.hpo import EstadoTrial


def estado_del_motor(
    estado_backend: EstadoTrial, atributos: Mapping[str, Any]
) -> EstadoTrial:
    estado = atributos.get("estado_motor", estado_backend)
    if estado not in ("pendiente", "corriendo", "completado", "podado", "fallido"):
        raise ValueError(f"estado_motor desconocido: {estado!r}")
    return estado
