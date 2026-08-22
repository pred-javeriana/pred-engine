"""Error de barrera de esquema (fail-fast)."""

from __future__ import annotations


class SchemaBarrierError(ValueError):
    """El marco alineado viola el contrato matematico PRED."""

    def __init__(
        self,
        mensaje: str,
        *,
        row_index: int | None = None,
        column: str | None = None,
        raw_value: object = None,
    ) -> None:
        self.row_index = row_index
        self.column = column
        self.raw_value = raw_value
        super().__init__(mensaje)
