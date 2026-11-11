"""Errores tipados del motor de seleccion (fail-closed, sin PII)."""

from __future__ import annotations

from pred_engine.comun.hechos_error import ErrorConHechos


class SelectionError(ValueError):
    """Frontera 2.1/2.2: la seleccion no puede continuar."""


class SelectionContractError(SelectionError):
    """Un contrato de entrada, resultado o familia es invalido."""


class NingunTrialCompletoError(ErrorConHechos, SelectionContractError):
    """Ningun ensayo de HPO termino para el SKU (todos fallidos o podados).

    ``hechos`` lleva ``search``: ensayos fallidos y podados, y cada causa con un
    ejemplo.
    """


class UnregisteredFamilyError(SelectionError):
    """La politica pidio una familia sin estrategia registrada."""


class DuplicateStrategyError(SelectionError):
    """Se intento registrar una familia ya ocupada sin reemplazo explicito."""


class UnknownSkuClassError(SelectionError):
    """sku_class no pertenece al conjunto Syntetos-Boylan del Modulo 1."""


class RouterConfigurationError(SelectionError):
    """Dependencias del enrutador ausentes, vacias o de tipo incorrecto."""
