"""Hechos de un error del motor en campos (contrato frontend-motor 1.3.0, 11.1).

El texto de un error (``NombreDeClase: mensaje``) sirve para soporte, pero la
interfaz no debe leerlo. Una excepcion que conoce los hechos de su fallo los lleva
en ``hechos``, con la forma de ``ErrorFacts`` del contrato: ``search`` (busqueda
sin ensayos completos), ``fit`` (ajuste que fallo) e ``history`` (historia que
falto). ``hechos_de`` los lee de cualquier excepcion para escribirlos junto a
``code``.

Los nombres de los campos son los del contrato (en ingles, como el resto del
documento); los identificadores de causa y de razon son los del motor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

Hechos = dict[str, Any]


class ErrorConHechos(Exception):
    """Base de las excepciones que llevan sus hechos en ``hechos``.

    ``hechos`` vive en ``__dict__``, que ``pickle`` conserva: llegan intactos desde
    los procesos hijos de la ejecucion paralela.
    """

    def __init__(self, *args: object, hechos: Mapping[str, Any] | None = None) -> None:
        super().__init__(*args)
        self.hechos: Hechos | None = compactar(hechos) if hechos else None


def compactar(hechos: Mapping[str, Any]) -> Hechos:
    """``hechos`` sin valores nulos, sin numeros no finitos y sin grupos vacios.

    JSON no admite ``NaN`` ni infinitos, y un campo sin dato se omite en vez de viajar
    como ``null``.
    """
    limpio: Hechos = {}
    for clave, valor in hechos.items():
        if isinstance(valor, Mapping):
            valor = compactar(valor)
            if not valor:
                continue
        elif isinstance(valor, list | tuple):
            valor = [compactar(v) if isinstance(v, Mapping) else v for v in valor]
        elif isinstance(valor, float) and not math.isfinite(valor):
            continue
        elif valor is None:
            continue
        limpio[clave] = valor
    return limpio


def ajuste(
    razon: str,
    *,
    modelo: str | None = None,
    valor: float | None = None,
    limite: float | None = None,
) -> Hechos:
    """``{"fit": ...}``: por que un modelo no se ajusto o no pronostico."""
    return compactar(
        {"fit": {"model": modelo, "reason": razon, "value": valor, "limit": limite}}
    )


def historia(requerido: int, disponible: int) -> Hechos:
    """``{"history": ...}``: observaciones exigidas y disponibles."""
    return {"history": {"required": requerido, "available": disponible}}


def hechos_de(exc: BaseException) -> Hechos | None:
    """Los hechos de ``exc``; si no los conoce, los de su causa; si no, ``None``.

    Una excepcion que envuelve a otra (``raise ... from exc``), como
    ``PipelineExecutionError`` o un ``SelectionContractError`` por una serie corta,
    conserva los hechos de la causa real.
    """
    hechos = getattr(exc, "hechos", None)
    if isinstance(hechos, Mapping) and hechos:
        return compactar(hechos)
    causa = exc.__cause__
    return None if causa is None else hechos_de(causa)
