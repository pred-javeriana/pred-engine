"""Errores de la particion causal y de la evaluacion Walk-Forward."""

from __future__ import annotations


class VentanaInsuficienteError(ValueError):
    def __init__(self, *, requerido: int, disponible: int) -> None:
        self.requerido = requerido
        self.disponible = disponible
        super().__init__(
            f"se requieren al menos {requerido} observaciones, hay {disponible} "
            f"(deficit={requerido - disponible})"
        )


class EvaluacionError(RuntimeError):
    pass
