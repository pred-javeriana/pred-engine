"""Adaptador DL del contrato comun del router; no cambia la politica de familias."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda
from pred_engine.optimizacion.optimizadores.modelos_deep_learning.dl_selection import (
    FAMILIA_DL,
    seleccionar_configuracion_dl,
)
from pred_engine.optimizacion.optimizadores.modelos_deep_learning.espacio_dl import (
    EspacioDL,
)
from pred_engine.optimizacion.router.contratos import (
    PredictorFamily,
    SelectionRequest,
    SelectionResult,
    TopologicalProfile,
)
from pred_engine.optimizacion.router.errores import SelectionContractError


class DLSelectionStrategy:
    family: PredictorFamily = FAMILIA_DL

    def __init__(
        self,
        *,
        espacio: EspacioDL | None = None,
        n_trials: int = 12,
        horizonte: int = 7,
        paso: int = 7,
        min_train: int | None = None,
        metrica_objetivo: str = "mase",
        estacionalidad: int = 1,
        reglas: ReglasPoda | None = None,
        seed: int = 0,
        raiz_corrida: str | Path | None = None,
        sesion: str | None = None,
    ) -> None:
        if raiz_corrida is not None and (not sesion or not sesion.strip()):
            raise ValueError("sesion es obligatoria cuando se indica raiz_corrida")
        self._espacio = espacio
        self._n_trials = n_trials
        self._horizonte = horizonte
        self._paso = paso
        self._min_train = min_train
        self._metrica = metrica_objetivo
        self._estacionalidad = estacionalidad
        self._reglas = reglas
        self._seed = seed
        self._raiz_corrida = raiz_corrida
        self._sesion = sesion

    def select(
        self, request: SelectionRequest, profile: TopologicalProfile
    ) -> SelectionResult:
        if not request.series:
            raise SelectionContractError(f"sku_id={request.sku_id!r} no trae serie")
        ordenadas = sorted(request.series, key=lambda obs: obs.timestamp)
        serie = np.array([obs.demand_qty for obs in ordenadas], dtype=float)
        run_id = None
        if self._raiz_corrida is not None:
            # SKU/sesion son identidad, nunca segmentos de un path arbitrario.
            identidad = json.dumps([request.sku_id, self._sesion]).encode("utf-8")
            run_id = f"dl-{hashlib.sha256(identidad).hexdigest()}"
        try:
            estudio = seleccionar_configuracion_dl(
                serie,
                sku_id=request.sku_id,
                espacio=self._espacio,
                n_trials=self._n_trials,
                horizonte=self._horizonte,
                paso=self._paso,
                min_train=self._min_train,
                metrica_objetivo=self._metrica,
                estacionalidad=self._estacionalidad,
                reglas=self._reglas,
                seed=self._seed,
                raiz_corrida=self._raiz_corrida,
                run_id=run_id,
            )
        except ValueError as exc:
            raise SelectionContractError(
                f"seleccion DL imposible para sku_id={request.sku_id!r}: {exc}"
            ) from exc
        mejor = estudio.mejor
        if mejor is None or mejor.valor is None:
            raise SelectionContractError(
                f"ningun trial de HPO completo para sku_id={request.sku_id!r} "
                f"({estudio.n_fallidos} fallidos, {estudio.n_podados} podados)"
            )
        return SelectionResult(
            sku_id=request.sku_id,
            sku_class=request.sku_class,
            family=FAMILIA_DL,
            profile=profile,
            produced_by=type(self).__name__,
            payload={
                "hiperparametros": dict(mejor.configuracion),
                "metrica_objetivo": estudio.metrica_objetivo,
                "valor": mejor.valor,
                "n_ventanas": mejor.n_ventanas,
                "n_trials": len(estudio.trials),
                "n_completados": estudio.n_completados,
                "n_podados": estudio.n_podados,
                "n_fallidos": estudio.n_fallidos,
                "seed": estudio.seed,
                "estudio": estudio,
            },
        )
