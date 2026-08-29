"""Errores de la Fase 0 de simulacion de pre-ingesta (modulos 0.3 y 0.4)."""

from __future__ import annotations

from pathlib import Path


class PhysicalConstraintError(ValueError):
    """Un vector sintetico no puede cumplir una ley de conservacion fisica."""


class DivergenceRejectionExhausted(RuntimeError):
    """El bucle de rechazo agoto los reintentos sin obtener una serie valida."""

    def __init__(self, intentos: int, tolerancia: float) -> None:
        self.intentos = intentos
        self.tolerancia = tolerancia
        super().__init__(
            "Rejection sampling agotado tras "
            f"{intentos} reintentos con tolerancia {tolerancia:.4f}"
        )


class SchemaConformanceError(ValueError):
    """El artefacto consolidado no cumple el contrato de datos de la seccion 0.4."""

    def __init__(self, mensaje: str, *, fallas: tuple[str, ...] = ()) -> None:
        self.fallas = fallas
        super().__init__(mensaje)


class WormOverwriteError(FileExistsError):
    """Se intento sobrescribir un artefacto ya depositado en el directorio crudo."""

    def __init__(self, destino: Path | str) -> None:
        self.destino = Path(destino)
        super().__init__(
            "Politica WORM: el artefacto ya existe y no puede reescribirse: "
            f"{self.destino}"
        )
