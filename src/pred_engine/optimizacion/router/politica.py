"""Politica topológica inicial (matriz 2.2, versionada y determinista)."""

from __future__ import annotations

from collections.abc import Mapping

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos import SKU_CLASSES, SkuClass
from pred_engine.optimizacion.router.contratos import (
    PREDICTOR_FAMILIES,
    TOPOLOGICAL_PROFILES,
    PredictorFamily,
    RoutingDecision,
    TopologicalProfile,
)
from pred_engine.optimizacion.router.errores import UnknownSkuClassError

_logger = get_logger(__name__)

INITIAL_POLICY_VERSION = "2.2.0-initial"

PROFILE_BY_SKU_CLASS: Mapping[SkuClass, TopologicalProfile] = {
    "smooth": "dense_stable",
    "erratic": "dense_variable",
    "intermittent": "sparse_stable",
    "lumpy": "sparse_variable",
}

FAMILIES_BY_SKU_CLASS: Mapping[SkuClass, tuple[PredictorFamily, ...]] = {
    "smooth": ("classical", "ml", "dl", "foundation"),
    "erratic": ("classical", "ml", "dl", "foundation"),
    "intermittent": ("classical", "ml", "foundation"),
    "lumpy": ("classical", "foundation"),
}


def _validar_matriz() -> None:
    # La matriz debe cubrir exactamente las cuatro clases del contrato 1.4.
    for clase in SKU_CLASSES:
        if clase not in PROFILE_BY_SKU_CLASS or clase not in FAMILIES_BY_SKU_CLASS:
            raise RuntimeError(f"matriz incompleta para {clase!r}")
        perfil = PROFILE_BY_SKU_CLASS[clase]
        if perfil not in TOPOLOGICAL_PROFILES:
            raise RuntimeError(f"perfil invalido para {clase!r}")
        familias = FAMILIES_BY_SKU_CLASS[clase]
        if not familias:
            raise RuntimeError(f"sin familias para {clase!r}")
        if len(set(familias)) != len(familias):
            raise RuntimeError(f"familias duplicadas para {clase!r}")
        for familia in familias:
            if familia not in PREDICTOR_FAMILIES:
                raise RuntimeError(f"familia invalida {familia!r}")


_validar_matriz()


class InitialTopologyPolicy:
    """Materializa la matriz aprobada. Determinista para una misma version."""

    version: str = INITIAL_POLICY_VERSION

    def decide(self, sku_class: SkuClass) -> tuple[RoutingDecision, ...]:
        if sku_class not in FAMILIES_BY_SKU_CLASS:
            _logger.error("sku_class desconocida para la politica: %s", sku_class)
            raise UnknownSkuClassError(f"sku_class no soportada: {sku_class!r}")
        perfil = PROFILE_BY_SKU_CLASS[sku_class]
        decisiones = tuple(
            RoutingDecision(family=familia, profile=perfil)
            for familia in FAMILIES_BY_SKU_CLASS[sku_class]
        )
        _logger.info(
            "Politica %s resolvio clase=%s n_decisiones=%s",
            self.version,
            sku_class,
            len(decisiones),
        )
        return decisiones
