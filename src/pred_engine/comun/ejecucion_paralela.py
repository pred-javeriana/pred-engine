"""Ejecucion de unidades de trabajo independientes con aislamiento de fallos (3.0).

Una unidad (por ejemplo, SKU x familia) que falla no detiene a las demas: su
error queda como resultado de esa unidad y el resto sigue. Cada resultado guarda
el proceso, el inicio y el fin de la unidad, que son la evidencia observable de
que el trabajo corrio en paralelo.

Con ``procesos=1`` todo corre en el proceso actual, con el mismo aislamiento.
Con ``procesos > 1`` los hijos se crean con ``spawn`` (ver ``CONTEXTO_MP``), asi
que la funcion y las tareas deben poder serializarse con ``pickle``.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

from pred_engine.comun.logger import get_logger

_logger = get_logger(__name__)

EstadoUnidad = Literal["completada", "fallida"]

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
CONTEXTO_MP = multiprocessing.get_context("spawn")


def fijar_una_hebra_blas() -> None:
    for variable in _VARS_HILOS_BLAS:
        os.environ[variable] = "1"


@contextmanager
def blas_de_una_hebra() -> Iterator[None]:
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
    fijar_una_hebra_blas()
    try:
        yield
    finally:
        for variable, valor in previos.items():
            if valor is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = valor


@dataclass(frozen=True, slots=True)
class RegistroUnidad:
    """Traza de una unidad: estado, proceso y reloj de pared (epoch, segundos)."""

    estado: EstadoUnidad
    pid: int
    inicio: float
    fin: float
    error: str | None = None

    @property
    def duracion_s(self) -> float:
        return self.fin - self.inicio


@dataclass(frozen=True, slots=True, eq=False)
class ResultadoUnidad[R]:
    registro: RegistroUnidad
    valor: R | None = None


def _ejecutar_aislada[T, R](funcion: Callable[[T], R], tarea: T) -> ResultadoUnidad[R]:
    """Unidad de trabajo de nivel de modulo (``pickle`` no serializa closures)."""
    inicio = time.time()
    try:
        valor = funcion(tarea)
    except Exception as exc:
        registro = RegistroUnidad(
            "fallida", os.getpid(), inicio, time.time(), describir_error(exc)
        )
        return ResultadoUnidad(registro)
    return ResultadoUnidad(
        RegistroUnidad("completada", os.getpid(), inicio, time.time()), valor
    )


def describir_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def ejecutar_unidades[T, R](
    funcion: Callable[[T], R],
    tareas: Sequence[T],
    *,
    procesos: int = 1,
) -> list[ResultadoUnidad[R]]:
    """Ejecuta ``funcion(tarea)`` por tarea y devuelve los resultados en su orden.

    Una excepcion dentro de una unidad se registra como unidad ``fallida`` y no
    interrumpe a las demas. Si un proceso hijo muere (por ejemplo, por falta de
    memoria), el pool queda inservible: las unidades que no alcanzaron a
    terminar se registran como fallidas con esa causa, y una nueva ejecucion
    puede retomarlas.
    """
    if type(procesos) is not int or procesos < 1:
        raise ValueError(f"procesos debe ser un entero >= 1; se recibio {procesos!r}")
    if procesos == 1 or len(tareas) < 2:
        return [_ejecutar_aislada(funcion, tarea) for tarea in tareas]

    efectivos = min(procesos, len(tareas))
    _logger.info(
        "Ejecucion en paralelo | unidades=%d | procesos=%d", len(tareas), efectivos
    )
    resultados: dict[int, ResultadoUnidad[R]] = {}
    with (
        blas_de_una_hebra(),
        ProcessPoolExecutor(
            max_workers=efectivos,
            mp_context=CONTEXTO_MP,
            initializer=fijar_una_hebra_blas,
        ) as ejecutor,
    ):
        envio = time.time()
        futuros = {
            ejecutor.submit(_ejecutar_aislada, funcion, tarea): indice
            for indice, tarea in enumerate(tareas)
        }
        try:
            for futuro in as_completed(futuros):
                indice = futuros[futuro]
                try:
                    resultados[indice] = futuro.result()
                except Exception as exc:
                    _logger.error("Unidad %d sin resultado: %s", indice, exc)
                    registro = RegistroUnidad(
                        "fallida", 0, envio, time.time(), describir_error(exc)
                    )
                    resultados[indice] = ResultadoUnidad(registro)
        except BaseException:
            # Una interrupcion no debe dejar corriendo las unidades no iniciadas.
            for pendiente in futuros:
                pendiente.cancel()
            raise
    return [resultados[indice] for indice in range(len(tareas))]


def resumen_paralelismo(registros: Sequence[RegistroUnidad]) -> dict[str, Any]:
    """Evidencia de concurrencia real a partir de las trazas de las unidades.

    ``concurrencia_maxima`` cuenta cuantas unidades estuvieron en curso a la
    vez; ``aceleracion`` compara el tiempo de computo sumado con el tiempo de
    pared del tramo (1.0 equivale a ejecucion secuencial).
    """
    if not registros:
        return {
            "unidades": 0,
            "procesos_distintos": 0,
            "concurrencia_maxima": 0,
            "computo_s": 0.0,
            "pared_s": 0.0,
            "aceleracion": None,
        }
    eventos = sorted(
        [(r.inicio, 1) for r in registros] + [(r.fin, -1) for r in registros],
        key=lambda evento: (evento[0], evento[1]),
    )
    en_curso = maxima = 0
    for _, delta in eventos:
        en_curso += delta
        maxima = max(maxima, en_curso)
    computo = sum(r.duracion_s for r in registros)
    pared = max(r.fin for r in registros) - min(r.inicio for r in registros)
    return {
        "unidades": len(registros),
        "procesos_distintos": len({r.pid for r in registros}),
        "concurrencia_maxima": maxima,
        "computo_s": round(computo, 3),
        "pared_s": round(pared, 3),
        "aceleracion": round(computo / pared, 2) if pared > 0 else None,
    }
