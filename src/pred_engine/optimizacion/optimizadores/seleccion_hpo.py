"""Nucleo comun de seleccion por HPO: lo que clasicos, ML y DL comparten.

Las tres familias hacen lo mismo alrededor de `ejecutar_estudio`: validar la
serie, correr un estudio por SKU y, en modo panel, repartir SKUs entre
procesos. Lo que cambia por familia (espacio, fabrica, traduccion del
ganador) queda en cada modulo; aqui NO hay logica de TPE/ASHA -- eso vive en
`HPO/estudio.py`.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pred_engine.comun.dataclasses.hpo import ResultadoEstudio
from pred_engine.comun.logger import get_logger
from pred_engine.comun.walkforward.protocolos import FabricaPronosticador
from pred_engine.optimizacion.optimizadores.HPO.contratos import InfoTrial
from pred_engine.optimizacion.optimizadores.HPO.espacio import EspacioBusqueda
from pred_engine.optimizacion.optimizadores.HPO.estudio import ejecutar_estudio
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda

_logger = get_logger(__name__)


def exigir_serie(
    y: np.ndarray, *, min_train: int, horizonte: int, sku_id: str | None
) -> np.ndarray:
    """Serie 1D de float con longitud suficiente para al menos una ventana."""
    serie = np.asarray(y, dtype=float)
    if serie.ndim != 1:
        raise ValueError("y debe ser un array 1D")
    minimo_requerido = min_train + horizonte
    if len(serie) < minimo_requerido:
        raise ValueError(
            f"serie de longitud {len(serie)} insuficiente para SKU={sku_id!r}: "
            f"se requieren al menos {minimo_requerido} observaciones "
            f"(min_train={min_train} + horizonte={horizonte})"
        )
    return serie


def seleccionar_con_hpo(
    serie: np.ndarray,
    espacio: EspacioBusqueda,
    fabrica: FabricaPronosticador,
    *,
    familia: str,
    sku_id: str | None,
    n_trials: int,
    min_train: int,
    horizonte: int,
    paso: int,
    metrica_objetivo: str,
    estacionalidad: int,
    muestreador: Any | None = None,
    reglas: ReglasPoda | None = None,
    seed: int = 0,
    raiz_corrida: str | Path | None = None,
    run_id: str | None = None,
    historico_previo: Sequence[InfoTrial] | None = None,
) -> ResultadoEstudio:
    """Un estudio de HPO por SKU: TPE + ASHA + Walk-Forward, sin logica propia."""
    return ejecutar_estudio(
        serie,
        espacio,
        fabrica,
        n_trials=n_trials,
        min_train=min_train,
        horizonte=horizonte,
        paso=paso,
        metrica_objetivo=metrica_objetivo,
        estacionalidad=estacionalidad,
        muestreador=muestreador,
        reglas=reglas,
        seed=seed,
        familia=familia,
        sku_id=sku_id,
        raiz_corrida=raiz_corrida,
        run_id=run_id,
        historico_previo=historico_previo,
    )


def series_por_sku(panel: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    """Extrae una serie 1D por SKU, en orden alfabetico de `sku_id`.

    Separado del reparto porque el modo paralelo necesita materializar TODAS
    las series antes de repartirlas: lo que viaja a cada proceso es el array
    de NumPy, nunca el `DataFrame` (mucho mas caro de serializar). El orden
    alfabetico fija el orden del diccionario de salida y, con el, la
    reproducibilidad de logs e identificadores de trial.
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


def seleccionar_panel[R](
    panel: pd.DataFrame,
    seleccionar_una: Callable[..., R],
    *,
    n_procesos: int | None = None,
    **kwargs: Any,
) -> dict[str, R]:
    """Un estudio independiente por SKU: `seleccionar_una(serie, sku_id=..., **kwargs)`.

    `n_procesos=None` (o `1`) ejecuta en el proceso actual. Con `n_procesos > 1`
    los SKU se reparten entre procesos: los estudios no comparten NINGUN estado
    (serie, `Study`, `DecisorASHA` y podador son propios de cada uno), asi que el
    reparto es seguro y el resultado es identico al secuencial --
    `derivar_semilla` es un hash de coordenadas estables, no un contador.

    `seleccionar_una` debe ser un invocable de nivel de modulo (se serializa con
    `pickle`). Los hijos se crean con `spawn` (ver `_CONTEXTO_MP`) y reimportan
    el modulo, asi que el script que llame con `n_procesos > 1` DEBE protegerse
    con `if __name__ == "__main__":`.
    """
    if n_procesos is not None and n_procesos < 1:
        raise ValueError(
            "n_procesos debe ser >= 1 o None (modo secuencial); "
            f"se recibio {n_procesos}"
        )

    series = series_por_sku(panel)
    if n_procesos is None or n_procesos == 1 or len(series) < 2:
        return {
            sku_id: seleccionar_una(serie, sku_id=sku_id, **kwargs)
            for sku_id, serie in series
        }

    return _seleccionar_en_paralelo(
        series, seleccionar_una, n_procesos=n_procesos, kwargs=kwargs
    )


def _trabajo_sku[R](
    seleccionar_una: Callable[..., R],
    sku_id: str,
    serie: np.ndarray,
    kwargs: dict[str, Any],
) -> tuple[str, R]:
    """Unidad de trabajo de un proceso hijo.

    Vive a nivel de modulo (y no como closure) porque `ProcessPoolExecutor`
    serializa el invocable con `pickle`, y `pickle` no serializa funciones
    anidadas. Devuelve el `sku_id` junto al resultado para poder reordenar la
    salida: los futuros se resuelven en orden de terminacion, no de envio.
    """
    return sku_id, seleccionar_una(serie, sku_id=sku_id, **kwargs)


_VARS_HILOS_BLAS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


# `spawn` y no `fork`: LightGBM (OpenMP/libgomp) se cuelga en el hijo si el padre
# ya entreno un modelo antes de bifurcarse (el runtime de OpenMP no sobrevive a
# `fork`). `spawn` arranca un interprete limpio por proceso, igual en todos los
# sistemas operativos, a costa de reimportar el paquete al iniciar el pool.
_CONTEXTO_MP = multiprocessing.get_context("spawn")


def _fijar_una_hebra_blas() -> None:
    for variable in _VARS_HILOS_BLAS:
        os.environ[variable] = "1"


@contextmanager
def _blas_de_una_hebra() -> Iterator[None]:
    """Evita la sobresuscripcion de hilos mientras vive el pool.

    NumPy/SciPy lanzan sus propios hilos de BLAS dentro de cada ajuste. Con N
    procesos x M hilos de BLAS el planificador del sistema operativo pasa mas
    tiempo cambiando de contexto que calculando, y el modo paralelo puede
    resultar MAS LENTO que el secuencial.

    Se fija en el proceso padre y no solo en el `initializer` porque con
    `spawn` el hijo hereda `os.environ` al arrancar, y las bibliotecas de
    BLAS leen estas variables UNA VEZ, al importarse. El entorno del padre se
    restaura al salir.
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


def _seleccionar_en_paralelo[R](
    series: list[tuple[str, np.ndarray]],
    seleccionar_una: Callable[..., R],
    *,
    n_procesos: int,
    kwargs: dict[str, Any],
) -> dict[str, R]:
    procesos = min(n_procesos, len(series))
    parciales: dict[str, R] = {}

    _logger.info("Seleccion en paralelo | skus=%d | procesos=%d", len(series), procesos)
    with (
        _blas_de_una_hebra(),
        ProcessPoolExecutor(
            max_workers=procesos,
            mp_context=_CONTEXTO_MP,
            initializer=_fijar_una_hebra_blas,
        ) as ejecutor,
    ):
        futuros = {
            ejecutor.submit(
                _trabajo_sku, seleccionar_una, sku_id, serie, kwargs
            ): sku_id
            for sku_id, serie in series
        }
        try:
            for futuro in as_completed(futuros):
                sku_id, resultado = futuro.result()
                parciales[sku_id] = resultado
        except BaseException:
            # Un SKU que falla no debe dejar corriendo los que aun no
            # arrancaron; los ya iniciados si se esperan al cerrar el pool.
            for pendiente in futuros:
                pendiente.cancel()
            raise

    # Se reconstruye en el orden de `series` (alfabetico por SKU) para que la
    # salida sea identica a la del modo secuencial, no la de terminacion.
    return {sku_id: parciales[sku_id] for sku_id, _ in series}
