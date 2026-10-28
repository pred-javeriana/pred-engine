"""Productos excluidos antes de L2 con su causa registrada."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SkuExclusion:
    """SKU descartado antes de L2 con su causa registrada (3.1)."""

    sku_id: str
    cause: str
    code: str | None = None
