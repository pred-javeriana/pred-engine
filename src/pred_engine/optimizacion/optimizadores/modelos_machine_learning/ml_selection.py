"""Seleccion de hiperparametros de ML mediante HPO sobre Walk-Forward Validation.

Thin wrapper (spec 2.5): define el espacio y la fabrica de LightGBM y delega
TODA la optimizacion en `seleccion_hpo.seleccionar_con_hpo` -> `ejecutar_estudio`
(TPE + ASHA + Walk-Forward). No hay bucle ask/tell ni Optuna en este archivo.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pred_engine.comun.dataclasses.modelos_machine_learning import (
    ConfiguracionSeleccionadaML,
    ResultadoSeleccionML,
)
from pred_engine.comun.modelos.modelos_machine_learning.lgbm import (
    LightGBMForecaster,
    fabrica_ml,
)
from pred_engine.optimizacion.optimizadores.HPO.contratos import InfoTrial
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda
from pred_engine.optimizacion.optimizadores.modelos_machine_learning.espacio_ml import (
    EspacioML,
    construir_espacio_ml,
    min_train_recomendado_ml,
)
from pred_engine.optimizacion.optimizadores.seleccion_hpo import (
    exigir_serie,
    seleccionar_con_hpo,
    seleccionar_panel,
)

FAMILIA_ML = "ml"


def seleccionar_configuracion_ml(
    y: np.ndarray,
    sku_id: str | None = None,
    espacio: EspacioML | None = None,
    n_trials: int = 30,
    min_train: int | None = None,
    horizonte: int = 7,
    paso: int = 7,
    metrica_objetivo: str = "mase",
    muestreador: Any | None = None,
    reglas: ReglasPoda | None = None,
    seed: int = 0,
    raiz_corrida: str | Path | None = None,
    run_id: str | None = None,
    historico_previo: Sequence[InfoTrial] | None = None,
) -> ResultadoSeleccionML:
    cfg = espacio or EspacioML()
    min_train_efectivo = (
        min_train if min_train is not None else min_train_recomendado_ml(cfg)
    )
    serie = exigir_serie(
        y, min_train=min_train_efectivo, horizonte=horizonte, sku_id=sku_id
    )
    m_efectivo = cfg.m if cfg.m > 1 else 1

    def _fabrica(configuracion: Mapping[str, Any], *, seed: int) -> LightGBMForecaster:
        completa = dict(configuracion)
        completa["m"] = m_efectivo
        return fabrica_ml(completa, seed=seed)

    resultado_estudio = seleccionar_con_hpo(
        serie,
        construir_espacio_ml(cfg),
        _fabrica,
        familia=FAMILIA_ML,
        sku_id=sku_id,
        n_trials=n_trials,
        min_train=min_train_efectivo,
        horizonte=horizonte,
        paso=paso,
        metrica_objetivo=metrica_objetivo,
        estacionalidad=m_efectivo,
        muestreador=muestreador,
        reglas=reglas,
        seed=seed,
        raiz_corrida=raiz_corrida,
        run_id=run_id,
        historico_previo=historico_previo,
    )

    seleccionada = None
    mejor = resultado_estudio.mejor
    if mejor is not None and mejor.valor is not None:
        seleccionada = ConfiguracionSeleccionadaML(
            sku_id=sku_id,
            hiperparametros=dict(mejor.configuracion),
            metrica_objetivo=metrica_objetivo,
            valor=mejor.valor,
            n_ventanas=mejor.n_ventanas,
        )
    return ResultadoSeleccionML(seleccionada=seleccionada, estudio=resultado_estudio)


def seleccionar_por_panel_ml(
    panel: pd.DataFrame,
    *,
    n_procesos: int | None = None,
    **kwargs: Any,
) -> dict[str, ResultadoSeleccionML]:
    """Un estudio de HPO independiente por SKU (ver `seleccionar_panel`)."""
    if "run_id" in kwargs:
        raise ValueError(
            "seleccionar_por_panel_ml no acepta 'run_id' (colisionaria entre "
            "SKUs, cada uno necesita el suyo); llama a "
            "seleccionar_configuracion_ml por SKU con un run_id propio "
            "si necesitas persistencia/reanudacion en modo panel"
        )
    return seleccionar_panel(
        panel, seleccionar_configuracion_ml, n_procesos=n_procesos, **kwargs
    )
