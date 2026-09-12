"""Seleccion de configuracion SARIMA mediante HPO sobre Walk-Forward Validation."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
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

    def _fabrica(configuracion: Mapping[str, Any], *, seed: int) -> SarimaForecaster:
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


def series_por_sku(panel: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    """Extrae una serie 1D por SKU, en orden alfabetico de `sku_id`.

    Separado de `seleccionar_por_panel` porque el modo paralelo necesita
    materializar TODAS las series antes de repartirlas: lo que viaja a cada
    proceso es el array de NumPy, nunca el `DataFrame` (mucho mas caro de
    serializar). El orden alfabetico fija el orden del diccionario de salida
    y, con el, la reproducibilidad de logs e identificadores de trial.
    """
    if "sku_id" not in panel.columns or "demand_qty" not in panel.columns:
        raise ValueError("panel debe tener columnas 'sku_id' y 'demand_qty'")

    columna_orden = "timestamp" if "timestamp" in panel.columns else None
    series: list[tuple[str, np.ndarray]] = []
    for sku_id, grupo in panel.groupby("sku_id", sort=True):
        if columna_orden is not None:
            grupo = grupo.sort_values(columna_orden)
        series.append((str(sku_id), grupo["demand_qty"].to_numpy(dtype=float)))
    return series


def seleccionar_por_panel(
    panel: pd.DataFrame,
    *,
    n_procesos: int | None = None,
    **kwargs: Any,
) -> dict[str, ResultadoSeleccionClasica]:
    """Un estudio de HPO independiente por SKU.

    `n_procesos=None` (o `1`) ejecuta en el proceso actual, igual que siempre.
    Con `n_procesos > 1` los SKU se reparten entre procesos: los estudios no
    comparten NINGUN estado (serie, `Study`, `DecisorASHA` y podador son
    propios de cada uno), asi que el reparto es seguro y el resultado es
    identico al secuencial -- `derivar_semilla` es un hash de coordenadas
    estables, no un contador, de modo que las semillas por ventana no
    dependen del orden de ejecucion.

    En Windows (metodo `spawn`) el proceso hijo reimporta este modulo, asi
    que el script que llame con `n_procesos > 1` DEBE protegerse con
    `if __name__ == "__main__":` o se replicara indefinidamente.
    """
    if n_procesos is not None and n_procesos < 1:
        raise ValueError(
            "n_procesos debe ser >= 1 o None (modo secuencial); "
            f"se recibio {n_procesos}"
        )

    series = series_por_sku(panel)
    if n_procesos is None or n_procesos == 1 or len(series) < 2:
        return {
            sku_id: seleccionar_configuracion_clasica(serie, sku_id=sku_id, **kwargs)
            for sku_id, serie in series
        }

    return _seleccionar_en_paralelo(series, n_procesos=n_procesos, kwargs=kwargs)


def _trabajo_sku(
    sku_id: str, serie: np.ndarray, kwargs: dict[str, Any]
) -> tuple[str, ResultadoSeleccionClasica]:
    """Unidad de trabajo de un proceso hijo.

    Vive a nivel de modulo (y no como closure dentro de
    `seleccionar_por_panel`) porque `ProcessPoolExecutor` serializa el
    invocable con `pickle`, y `pickle` no serializa funciones anidadas.
    Devuelve el `sku_id` junto al resultado para poder reordenar la salida:
    los futuros se resuelven en orden de terminacion, no de envio.
    """
    return sku_id, seleccionar_configuracion_clasica(serie, sku_id=sku_id, **kwargs)


_VARS_HILOS_BLAS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _fijar_una_hebra_blas() -> None:
    for variable in _VARS_HILOS_BLAS:
        os.environ[variable] = "1"


@contextmanager
def _blas_de_una_hebra() -> Iterator[None]:
    """Evita la sobresuscripcion de hilos mientras vive el pool.

    NumPy/SciPy lanzan sus propios hilos de BLAS dentro de cada ajuste de
    SARIMAX. Con N procesos x M hilos de BLAS el planificador del sistema
    operativo pasa mas tiempo cambiando de contexto que calculando, y el
    modo paralelo puede resultar MAS LENTO que el secuencial.

    Se fija en el proceso padre y no solo en el `initializer` porque con
    `spawn` el hijo hereda `os.environ` al arrancar, y las bibliotecas de
    BLAS leen estas variables UNA VEZ, al importarse: fijarlas despues del
    import no tendria efecto. El entorno del padre se restaura al salir.
    """
    previos = {variable: os.environ.get(variable) for variable in _VARS_HILOS_BLAS}
    _fijar_una_hebra_blas()
    try:
        yield
    finally:
        for variable, valor in previos.items():
            if valor is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = valor


def _seleccionar_en_paralelo(
    series: list[tuple[str, np.ndarray]],
    *,
    n_procesos: int,
    kwargs: dict[str, Any],
) -> dict[str, ResultadoSeleccionClasica]:
    procesos = min(n_procesos, len(series))
    parciales: dict[str, ResultadoSeleccionClasica] = {}

    _logger.info(
        "Seleccion clasica en paralelo | skus=%d | procesos=%d", len(series), procesos
    )
    with (
        _blas_de_una_hebra(),
        ProcessPoolExecutor(
            max_workers=procesos, initializer=_fijar_una_hebra_blas
        ) as ejecutor,
    ):
        futuros = {
            ejecutor.submit(_trabajo_sku, sku_id, serie, kwargs): sku_id
            for sku_id, serie in series
        }
        try:
            for futuro in as_completed(futuros):
                sku_id, resultado = futuro.result()
                parciales[sku_id] = resultado
        except BaseException:
            # Un SKU que falla no debe dejar corriendo los que aun no
            # arrancaron; los ya iniciados sí se esperan al cerrar el pool.
            for pendiente in futuros:
                pendiente.cancel()
            raise

    # Se reconstruye en el orden de `series` (alfabetico por SKU) para que la
    # salida sea identica a la del modo secuencial, no la de terminacion.
    return {sku_id: parciales[sku_id] for sku_id, _ in series}
