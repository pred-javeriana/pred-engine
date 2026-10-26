"""Resultados durables de L2/L3; JSON tipado, sin modelos ajustados ni pickle.

El llamador fija una identidad del experimento (codigo, versiones y opciones).
Cada clave agrega la entrada efectiva de la unidad. Solo se guardan exitos;
una unidad fallida o un archivo temporal nunca cuenta como trabajo terminado.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from pred_engine.comun.dataclasses.validacion_temporal import (
    ResultadoVentana,
    ResultadoWalkForward,
    VentanaTemporal,
)
from pred_engine.comun.ejecucion_paralela import RegistroUnidad, ResultadoUnidad
from pred_engine.optimizacion.control_reanudacion.almacenamiento import escribir_atomico
from pred_engine.optimizacion.router.contratos import SelectionResult


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _types() -> dict[str, type]:
    # La importacion tardia evita acoplar la construccion del pipeline al codec.
    from pred_engine.pipeline import FittedCandidate

    return {
        c.__name__: c
        for c in (
            FittedCandidate,
            ResultadoUnidad,
            RegistroUnidad,
            ResultadoWalkForward,
            ResultadoVentana,
            VentanaTemporal,
        )
    }


def _encode(value: Any) -> Any:
    if isinstance(value, SelectionResult):
        return {"type": "selection", "data": _encode(value.model_dump(mode="python"))}
    if isinstance(value, np.ndarray):
        return {"type": "array", "data": value.tolist()}
    if is_dataclass(value) and not isinstance(value, type):
        if type(value).__name__ not in _types():
            raise ValueError("tipo de checkpoint no permitido")
        return {
            "type": type(value).__name__,
            "data": {f.name: _encode(getattr(value, f.name)) for f in fields(value)},
        }
    if isinstance(value, Mapping):
        return {"type": "mapping", "data": {k: _encode(v) for k, v in value.items()}}
    if isinstance(value, tuple):
        return {"type": "tuple", "data": [_encode(v) for v in value]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"valor de checkpoint no soportado: {type(value).__name__}")


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    kind, data = value["type"], value["data"]
    if kind == "selection":
        return SelectionResult.model_validate(_decode(data))
    if kind == "array":
        return np.asarray(data, dtype=float)
    if kind == "mapping":
        return {k: _decode(v) for k, v in data.items()}
    if kind == "tuple":
        return tuple(_decode(v) for v in data)
    constructor = _types().get(kind)
    if constructor is None:
        raise ValueError("tipo de checkpoint no permitido")
    return constructor(**{k: _decode(v) for k, v in data.items()})


class UnitCheckpoints:
    """Almacen optativo; la identidad pertenece al experimento, no al intento."""

    def __init__(self, root: str | Path, *, identity: str) -> None:
        if not identity.strip():
            raise ValueError("el checkpoint requiere identidad del experimento")
        self.root, self.identity = Path(root), identity

    def path(self, stage: str, context: Any) -> Path:
        key = hashlib.sha256(
            _json([self.identity, stage, context]).encode()
        ).hexdigest()
        return self.root / stage / f"{key}.json"


def save_result(path: Path | None, result: ResultadoUnidad[Any]) -> None:
    if path is None or result.registro.estado != "completada" or result.valor is None:
        return
    data = {
        "schema": "pred-engine.unidad/1",
        "key": path.stem,
        "result": _encode(result),
    }
    envelope = {
        "data": data,
        "sha256": hashlib.sha256(_json(data).encode()).hexdigest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    escribir_atomico(path, _json(envelope) + "\n")
    # Windows requiere acceso de escritura para sincronizar el archivo a disco.
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())
    if os.name != "nt":
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def load_result(path: Path | None) -> ResultadoUnidad[Any] | None:
    if path is None or not path.exists():
        return None
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        data = envelope["data"]
        if (
            data["schema"] != "pred-engine.unidad/1"
            or data["key"] != path.stem
            or hashlib.sha256(_json(data).encode()).hexdigest() != envelope["sha256"]
        ):
            raise ValueError("identidad o integridad invalida")
        result = _decode(data["result"])
        if (
            not isinstance(result, ResultadoUnidad)
            or result.valor is None
            or result.registro.estado != "completada"
        ):
            raise ValueError("el checkpoint no contiene una unidad exitosa")
        return replace(result, reutilizado=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint invalido {path.name}: {exc}") from exc
