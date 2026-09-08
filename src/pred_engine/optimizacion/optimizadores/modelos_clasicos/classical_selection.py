"""Seleccion de configuracion SARIMA mediante HPO sobre Walk-Forward Validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd

from pred_engine.comun.dataclasses.modelos_clasicos import (
    ConfiguracionSeleccionada,
    ResultadoSeleccionClasica,
)
from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos.modelos_clasicos.sarima import SarimaForecaster
from pred_engine.optimizacion.optimizadores.HPO.espacio import Entero, EspacioBusqueda
from pred_engine.optimizacion.optimizadores.HPO.estudio import ejecutar_estudio
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EspacioClasico:
    p_max: int = 3
    d_max: int = 2
    q_max: int = 3
    P_max: int = 2
    D_max: int = 1
    Q_max: int = 2
    m: int = 7
    max_orden_total: int = 6


def construir_espacio(espacio: EspacioClasico | None = None) -> EspacioBusqueda:
    cfg = espacio or EspacioClasico()
    usa_estacionalidad = cfg.m > 1

    parametros: list[Entero] = [
        Entero("p", 0, cfg.p_max),
        Entero("d", 0, cfg.d_max),
        Entero("q", 0, cfg.q_max),
    ]
    if usa_estacionalidad:
        parametros += [
            Entero("P", 0, cfg.P_max),
            Entero("D", 0, cfg.D_max),
            Entero("Q", 0, cfg.Q_max),
        ]

    def _no_degenerada(configuracion: Mapping[str, Any]) -> bool:
        no_estacional_nula = (
            configuracion["p"] == configuracion["d"] == configuracion["q"] == 0
        )
        if not usa_estacionalidad:
            return not no_estacional_nula
        estacional_nula = (
            configuracion.get("P", 0) == 0
            and configuracion.get("D", 0) == 0
            and configuracion.get("Q", 0) == 0
        )
        return not (no_estacional_nula and estacional_nula)

    def _orden_total_acotado(configuracion: Mapping[str, Any]) -> bool:
        orden = (
            configuracion["p"]
            + configuracion["q"]
            + configuracion.get("P", 0)
            + configuracion.get("Q", 0)
        )
        return orden <= cfg.max_orden_total

    return EspacioBusqueda(
        parametros=tuple(parametros),
        restricciones=(_no_degenerada, _orden_total_acotado),
    )


def fabrica_sarima(configuracion: Mapping[str, Any], *, seed: int) -> SarimaForecaster:

    order = (configuracion["p"], configuracion["d"], configuracion["q"])
    seasonal_order = (
        configuracion.get("P", 0),
        configuracion.get("D", 0),
        configuracion.get("Q", 0),
        configuracion.get("m", 0),
    )
    return SarimaForecaster(order=order, seasonal_order=seasonal_order, seed=seed)


def min_train_recomendado(espacio: EspacioClasico | None = None) -> int:
    cfg = espacio or EspacioClasico()
    orden_no_estacional = cfg.p_max + cfg.d_max + cfg.q_max
    orden_estacional = cfg.P_max + cfg.D_max + cfg.Q_max
    base = 2 * cfg.m if cfg.m > 1 else 0
    return base + max(orden_no_estacional, orden_estacional, 10)


def seleccionar_configuracion_clasica(
    y: np.ndarray,
    sku_id: str | None = None,
    espacio: EspacioClasico | None = None,
    n_trials: int = 40,
    min_train: int | None = None,
    horizonte: int = 7,
    paso: int = 7,
    metrica_objetivo: str = "mase",
    muestreador: optuna.samplers.BaseSampler | None = None,
    reglas: ReglasPoda | None = None,
    seed: int = 0,
) -> ResultadoSeleccionClasica:
    cfg = espacio or EspacioClasico()
    serie = np.asarray(y, dtype=float)
    if serie.ndim != 1:
        raise ValueError("y debe ser un array 1D")

    min_train_efectivo = (
        min_train if min_train is not None else min_train_recomendado(cfg)
    )
    minimo_requerido = min_train_efectivo + horizonte
    if len(serie) < minimo_requerido:
        raise ValueError(
            f"serie de longitud {len(serie)} insuficiente para SKU={sku_id!r}: "
            f"se requieren al menos {minimo_requerido} observaciones "
            f"(min_train={min_train_efectivo} + horizonte={horizonte})"
        )

    espacio_hpo = construir_espacio(cfg)
    m_efectivo = cfg.m if cfg.m > 1 else 0

    def _fabrica(configuracion: Mapping[str, Any], seed: int) -> SarimaForecaster:
        configuracion_completa = dict(configuracion)
        configuracion_completa["m"] = m_efectivo
        return fabrica_sarima(configuracion_completa, seed=seed)

    resultado_estudio = ejecutar_estudio(
        serie,
        espacio_hpo,
        _fabrica,
        n_trials=n_trials,
        min_train=min_train_efectivo,
        horizonte=horizonte,
        paso=paso,
        metrica_objetivo=metrica_objetivo,
        estacionalidad=cfg.m if cfg.m > 1 else 1,
        muestreador=muestreador,
        reglas=reglas,
        seed=seed,
        familia="clasicos",
        sku_id=sku_id,
    )

    seleccionada = None
    if resultado_estudio.mejor is not None:
        mejor = resultado_estudio.mejor
        seleccionada = ConfiguracionSeleccionada(
            sku_id=sku_id,
            order=(
                mejor.configuracion["p"],
                mejor.configuracion["d"],
                mejor.configuracion["q"],
            ),
            seasonal_order=(
                mejor.configuracion.get("P", 0),
                mejor.configuracion.get("D", 0),
                mejor.configuracion.get("Q", 0),
                m_efectivo,
            ),
            metrica_objetivo=metrica_objetivo,
            valor=mejor.valor,  # type: ignore[arg-type]
            n_ventanas=mejor.n_ventanas,
        )

    return ResultadoSeleccionClasica(
        seleccionada=seleccionada, estudio=resultado_estudio
    )


def seleccionar_por_panel(
    panel: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, ResultadoSeleccionClasica]:
    if "sku_id" not in panel.columns or "demand_qty" not in panel.columns:
        raise ValueError("panel debe tener columnas 'sku_id' y 'demand_qty'")

    resultados: dict[str, ResultadoSeleccionClasica] = {}
    columna_orden = "timestamp" if "timestamp" in panel.columns else None

    for sku_id, grupo in panel.groupby("sku_id", sort=True):
        if columna_orden is not None:
            grupo = grupo.sort_values(columna_orden)
        serie = grupo["demand_qty"].to_numpy(dtype=float)
        resultados[str(sku_id)] = seleccionar_configuracion_clasica(
            serie, sku_id=str(sku_id), **kwargs
        )
    return resultados
