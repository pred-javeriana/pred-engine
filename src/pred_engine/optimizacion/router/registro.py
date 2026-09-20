"""Registro desacoplado de estrategias indexado por familia."""

from __future__ import annotations

from pred_engine.comun.logger import get_logger
from pred_engine.optimizacion.router.contratos import (
    PREDICTOR_FAMILIES,
    PredictorFamily,
    SelectionStrategy,
)
from pred_engine.optimizacion.router.errores import (
    DuplicateStrategyError,
    SelectionContractError,
    UnregisteredFamilyError,
)

_logger = get_logger(__name__)


class StrategyRegistry:
    """Indice familia -> estrategia. Independiente de la politica topológica."""

    def __init__(self) -> None:
        self._estrategias: dict[PredictorFamily, SelectionStrategy] = {}

    @property
    def families(self) -> frozenset[PredictorFamily]:
        return frozenset(self._estrategias)

    def register(
        self,
        family: PredictorFamily,
        strategy: SelectionStrategy,
        *,
        replace: bool = False,
    ) -> None:
        familia = _exigir_familia(family)
        if not isinstance(strategy, SelectionStrategy):
            raise SelectionContractError(
                "la estrategia no satisface el contrato SelectionStrategy"
            )
        if strategy.family != familia:
            raise SelectionContractError(
                "la estrategia declara family="
                f"{strategy.family!r} pero se registro como {familia!r}"
            )
        if familia in self._estrategias and not replace:
            raise DuplicateStrategyError(
                f"ya existe una estrategia para {familia!r}; use replace=True"
            )
        if familia in self._estrategias and replace:
            _logger.warning("Reemplazando estrategia de familia=%s", familia)
        self._estrategias[familia] = strategy
        _logger.info("Estrategia registrada familia=%s", familia)

    def resolve(self, family: PredictorFamily) -> SelectionStrategy:
        familia = _exigir_familia(family)
        try:
            return self._estrategias[familia]
        except KeyError:
            _logger.error("Familia sin estrategia registrada: %s", familia)
            raise UnregisteredFamilyError(
                f"no hay estrategia registrada para {familia!r}"
            ) from None


def _exigir_familia(family: object) -> PredictorFamily:
    if family not in PREDICTOR_FAMILIES:
        raise SelectionContractError(f"familia de predictores invalida: {family!r}")
    return family  # type: ignore[return-value]
