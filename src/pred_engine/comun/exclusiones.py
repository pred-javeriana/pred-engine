"""Productos excluidos antes de L2 o al cerrar L2, con su causa registrada."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SkuExclusion:
    """SKU descartado antes de L2 (3.1) o al cerrar L2 (ADR-023), con su causa."""

    sku_id: str
    cause: str
    code: str | None = None
