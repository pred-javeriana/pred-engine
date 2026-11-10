"""Productos excluidos antes de L2 o al cerrar L2, con su causa registrada."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SkuExclusion:
    """SKU descartado antes de L2 (3.1) o al cerrar L2 (ADR-023), con su causa.

    ``hechos`` son los datos de la causa en campos (contrato 1.3.0, 11.1), p. ej. la
    historia exigida y la disponible.
    """

    sku_id: str
    cause: str
    code: str | None = None
    hechos: Mapping[str, Any] | None = None

    def como_dict(self) -> dict[str, Any]:
        """Como la escriben ``corrida.json`` y el evento ``excluded``."""
        fila: dict[str, Any] = {
            "sku_id": self.sku_id,
            "cause": self.cause,
            "code": self.code,
        }
        if self.hechos:
            fila["facts"] = dict(self.hechos)
        return fila
