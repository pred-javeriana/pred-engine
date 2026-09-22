"""`ClassicalSelectionStrategy`: la familia clasica como estrategia del router.

Adapta `SelectionRequest` (serie por SKU + perfil topologico) a
`seleccionar_configuracion_clasica` y empaqueta el ganador en `SelectionResult`.
El router no importa este modulo: la estrategia se inyecta al registrarla.

`classical_selection.py` nacio en 2.4, antes del router (ADR-008), y hasta
ahora solo se invocaba como funcion; este envoltorio es lo que faltaba para
que la politica 2.2 pueda enrutar la familia `classical` en las cuatro clases
de SKU.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pred_engine.comun.logger import get_logger
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda
from pred_engine.optimizacion.optimizadores.modelos_clasicos.classical_selection import (  # noqa: E501
    EspacioClasico,
    seleccionar_configuracion_clasica,
)
from pred_engine.optimizacion.router.contratos import (
    PredictorFamily,
    SelectionRequest,
    SelectionResult,
    TopologicalProfile,
)
from pred_engine.optimizacion.router.errores import SelectionContractError

_logger = get_logger(__name__)

FAMILIA_CLASICA: PredictorFamily = "classical"


@dataclass(frozen=True, slots=True)
class PresupuestoClasico:
    n_trials: int
    reglas: ReglasPoda = field(default_factory=ReglasPoda)


# Mismo criterio que ML (ADR-014): mas trials donde la demanda es variable,
# menos donde es escasa. Valores iniciales versionados con la estrategia;
# inyectables para sobre-escribirlos.
PRESUPUESTO_POR_PERFIL: Mapping[TopologicalProfile, PresupuestoClasico] = {
    "dense_stable": PresupuestoClasico(n_trials=30),
    "dense_variable": PresupuestoClasico(n_trials=40),
    "sparse_stable": PresupuestoClasico(n_trials=25),
    "sparse_variable": PresupuestoClasico(n_trials=25),
}


class ClassicalSelectionStrategy:
    family: PredictorFamily = FAMILIA_CLASICA

    def __init__(
        self,
        *,
        espacio: EspacioClasico | None = None,
        presupuestos: Mapping[TopologicalProfile, PresupuestoClasico] | None = None,
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
            run_id = f"{FAMILIA_CLASICA}-{request.sku_id}-{self._sesion}"

        try:
            resultado = seleccionar_configuracion_clasica(
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
                f"seleccion clasica imposible para sku_id={request.sku_id!r}: {exc}"
            ) from exc

        seleccionada = resultado.seleccionada
        if seleccionada is None:
            raise SelectionContractError(
                f"ningun trial de HPO completo para sku_id={request.sku_id!r} "
                f"({resultado.estudio.n_fallidos} fallidos, "
                f"{resultado.estudio.n_podados} podados)"
            )

        _logger.info(
            "Seleccion clasica sku_id=%s perfil=%s %s=%.6f trials=%d podados=%d",
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
            family=FAMILIA_CLASICA,
            profile=profile,
            produced_by=type(self).__name__,
            payload={
                # Listas y no tuplas: el payload viaja a JSON en el Modulo 3.
                "order": list(seleccionada.order),
                "seasonal_order": list(seleccionada.seasonal_order),
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
