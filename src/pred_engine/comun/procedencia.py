"""Origen de datos: una ausencia no declara datos reales."""

from __future__ import annotations


def resolver_origen(conocida: bool | None, declarada: bool | None) -> bool | None:
    """Conserva una declaración previa y rechaza su sustitución."""
    if any(
        value is not None and type(value) is not bool for value in (conocida, declarada)
    ):
        raise ValueError("el origen debe ser booleano o nulo")
    if conocida is not None and declarada is not None and conocida != declarada:
        raise ValueError("el origen real o sintetico de una ingesta no puede cambiar")
    return conocida if declarada is None else declarada
