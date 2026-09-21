"""`MLSelectionStrategy`: la familia ML como estrategia del router (2.1/2.2).

Adapta `SelectionRequest` (serie por SKU + perfil topologico) a
`seleccionar_configuracion_ml` y empaqueta el ganador en `SelectionResult`.
El router no importa este modulo: la estrategia se inyecta al registrarla.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pred_engine.comun.logger import get_logger
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda
from pred_engine.optimizacion.optimizadores.modelos_machine_learning.espacio_ml import (
    EspacioML,
)
from pred_engine.optimizacion.optimizadores.modelos_machine_learning.ml_selection import (  # noqa: E501
    FAMILIA_ML,
    seleccionar_configuracion_ml,
)
from pred_engine.optimizacion.router.contratos import (
    PredictorFamily,
    SelectionRequest,
    SelectionResult,
    TopologicalProfile,
)
from pred_engine.optimizacion.router.errores import SelectionContractError

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PresupuestoHPO:
    n_trials: int
    reglas: ReglasPoda = field(default_factory=ReglasPoda)


# Mas trials donde la demanda es variable (hay mas que aprender); menos donde
# es escasa (pocas observaciones no justifican una busqueda amplia). Valores
# iniciales versionados con la estrategia; inyectables para sobre-escribirlos.
PRESUPUESTO_POR_PERFIL: Mapping[TopologicalProfile, PresupuestoHPO] = {
    "dense_stable": PresupuestoHPO(n_trials=30),
    "dense_variable": PresupuestoHPO(n_trials=40),
    "sparse_stable": PresupuestoHPO(n_trials=25),
    "sparse_variable": PresupuestoHPO(n_trials=25),
}


class MLSelectionStrategy:
    family: PredictorFamily = FAMILIA_ML

    def __init__(
        self,
        *,
        espacio: EspacioML | None = None,
        presupuestos: Mapping[TopologicalProfile, PresupuestoHPO] | None = None,
        horizonte: int = 7,
        paso: int = 7,
        metrica_objetivo: str = "mase",
        min_train: int | None = None,
        seed: int = 0,
        raiz_corrida: str | Path | None = None,
        sesion: str | None = None,
    ) -> None:
        if raiz_corrida is not None and not sesion:
            raise ValueError(
                "sesion es obligatoria cuando se indica raiz_corrida: el run_id "
                "debe ser estable entre la interrupcion y la reanudacion"
            )
        self._espacio = espacio
        self._presupuestos = presupuestos or PRESUPUESTO_POR_PERFIL
        self._horizonte = horizonte
        self._paso = paso
        self._metrica = metrica_objetivo
        self._min_train = min_train
        self._seed = seed
        self._raiz_corrida = raiz_corrida
        self._sesion = sesion

    def select(
        self, request: SelectionRequest, profile: TopologicalProfile
    ) -> SelectionResult:
        presupuesto = self._presupuestos.get(profile)
        if presupuesto is None:
            raise SelectionContractError(
                f"sin presupuesto de HPO para el perfil {profile!r}"
            )
        serie = _serie_de(request)

        run_id = None
        if self._raiz_corrida is not None:
            run_id = f"{FAMILIA_ML}-{request.sku_id}-{self._sesion}"

        try:
            resultado = seleccionar_configuracion_ml(
                serie,
                sku_id=request.sku_id,
                espacio=self._espacio,
                n_trials=presupuesto.n_trials,
                min_train=self._min_train,
                horizonte=self._horizonte,
                paso=self._paso,
                metrica_objetivo=self._metrica,
                reglas=presupuesto.reglas,
                seed=self._seed,
                raiz_corrida=self._raiz_corrida,
                run_id=run_id,
            )
        except ValueError as exc:
            # Serie corta u otro dato invalido: fail-closed con mensaje del origen.
            raise SelectionContractError(
                f"seleccion ML imposible para sku_id={request.sku_id!r}: {exc}"
            ) from exc

        seleccionada = resultado.seleccionada
        if seleccionada is None:
            raise SelectionContractError(
                f"ningun trial de HPO completo para sku_id={request.sku_id!r} "
                f"({resultado.estudio.n_fallidos} fallidos, "
                f"{resultado.estudio.n_podados} podados)"
            )

        _logger.info(
            "Seleccion ML sku_id=%s perfil=%s %s=%.6f trials=%d podados=%d",
            request.sku_id,
            profile,
            self._metrica,
            seleccionada.valor,
            len(resultado.estudio.trials),
            resultado.estudio.n_podados,
        )
        return SelectionResult(
            sku_id=request.sku_id,
            sku_class=request.sku_class,
            family=FAMILIA_ML,
            profile=profile,
            produced_by=type(self).__name__,
            payload={
                "hiperparametros": dict(seleccionada.hiperparametros),
                "metrica_objetivo": seleccionada.metrica_objetivo,
                "valor": seleccionada.valor,
                "n_ventanas": seleccionada.n_ventanas,
                "n_trials": len(resultado.estudio.trials),
                "n_completados": resultado.estudio.n_completados,
                "n_podados": resultado.estudio.n_podados,
                "n_fallidos": resultado.estudio.n_fallidos,
                "seed": resultado.estudio.seed,
            },
        )


def _serie_de(request: SelectionRequest) -> np.ndarray:
    if not request.series:
        raise SelectionContractError(
            f"la solicitud de sku_id={request.sku_id!r} no trae serie"
        )
    ordenadas = sorted(request.series, key=lambda obs: obs.timestamp)
    return np.array([obs.demand_qty for obs in ordenadas], dtype=float)
