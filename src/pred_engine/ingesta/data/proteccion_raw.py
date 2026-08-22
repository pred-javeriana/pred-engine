"""Guardia de solo lectura sobre el almacenamiento crudo inmutable."""

from __future__ import annotations

import builtins
import inspect
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from pred_engine.comun.logger import get_logger

P = ParamSpec("P")
R = TypeVar("R")

_MODOS_ESCRITURA = frozenset({"w", "a", "x", "+"})


class RawWritePermissionError(PermissionError):
    """Violacion de la politica de solo lectura sobre /raw."""

    def __init__(self, path: Path | str, caller: str) -> None:
        self.path = Path(path)
        self.caller = caller
        super().__init__(
            "Escritura bloqueada en almacenamiento crudo inmutable: "
            f"{self.path} (llamador: {caller})"
        )


def _es_modo_escritura(mode: str) -> bool:
    return any(flag in mode for flag in _MODOS_ESCRITURA)


def _como_ruta(file: Any) -> Path | None:
    if isinstance(file, (str, os.PathLike)):
        return Path(file)
    return None


def _esta_bajo(ruta: Path, raiz: Path) -> bool:
    try:
        ruta.expanduser().resolve().relative_to(raiz.resolve())
        return True
    except (ValueError, OSError):
        return False


def _nombre_llamador() -> str:
    modulo_actual = Path(__file__).resolve()
    for frame_info in inspect.stack()[1:]:
        archivo = Path(frame_info.filename).resolve()
        if archivo != modulo_actual:
            return frame_info.function
    return "<desconocido>"


def _resolver_modo(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    if "mode" in kwargs:
        return str(kwargs["mode"])
    if args:
        return str(args[0])
    return "r"


@contextmanager
def raw_read_only_guard(raw_root: str | Path) -> Iterator[None]:
    """Intercepta open() en modo escritura bajo `raw_root` y luego restaura."""
    raiz = Path(raw_root).resolve()
    logger = get_logger(__name__)
    original_open = builtins.open

    def guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        modo = _resolver_modo(args, kwargs)
        ruta = _como_ruta(file)
        if (
            ruta is not None
            and _es_modo_escritura(modo)
            and _esta_bajo(ruta, raiz)
        ):
            llamador = _nombre_llamador()
            error = RawWritePermissionError(ruta, llamador)
            logger.error("%s", error)
            raise error
        return original_open(file, *args, **kwargs)

    builtins.open = guarded_open  # type: ignore[assignment]
    try:
        yield
    finally:
        builtins.open = original_open


def enforce_raw_read_only(
    raw_root: str | Path,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorador que ejecuta la funcion bajo la guardia de solo lectura."""

    def decorador(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def envoltorio(*args: P.args, **kwargs: P.kwargs) -> R:
            with raw_read_only_guard(raw_root):
                return func(*args, **kwargs)

        return envoltorio

    return decorador
