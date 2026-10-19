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

import json
import multiprocessing
import os
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pred_engine.comun.errores_codigo import codigo_de
from pred_engine.comun.logger import get_logger

_logger = get_logger(__name__)

EstadoUnidad = Literal["completada", "fallida"]

# Formato del flujo de progreso de una corrida (``progreso.jsonl``): una linea JSON por
# evento, que la plataforma lee mientras la corrida esta en curso (G3).
ESQUEMA_PROGRESO = "pred-engine.progreso/1"

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
    """Traza de una unidad: estado, proceso y reloj de pared (epoch, segundos).

    ``cpu_s`` es el tiempo de CPU que consumio el proceso mientras corria la
    unidad (todas sus hebras); ``None`` si la unidad no llego a correr. ``codigo`` es
    el codigo estable del error (seccion 11 del contrato), solo en unidades fallidas.
    """

    estado: EstadoUnidad
    pid: int
    inicio: float
    fin: float
    error: str | None = None
    cpu_s: float | None = None
    codigo: str | None = None

    @property
    def duracion_s(self) -> float:
        return self.fin - self.inicio


@dataclass(frozen=True, slots=True, eq=False)
class ResultadoUnidad[R]:
    registro: RegistroUnidad
    valor: R | None = None


@dataclass(frozen=True, slots=True)
class EtiquetaUnidad:
    """Identidad observable de una unidad en el flujo de progreso."""

    etapa: str
    sku_id: str
    family: str
    model: str

    def como_dict(self) -> dict[str, str]:
        return {
            "stage": self.etapa,
            "sku_id": self.sku_id,
            "family": self.family,
            "model": self.model,
        }


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).isoformat()


def emitir_evento(ruta: str | Path, evento: str, /, **datos: Any) -> None:
    """Agrega una linea JSON a ``ruta`` con una sola escritura ``O_APPEND``.

    El progreso es observabilidad: un fallo al escribirlo se registra y nunca
    interrumpe el trabajo.
    """
    registro = {"event": evento, "t": _iso(time.time()), **datos}
    try:
        linea = json.dumps(registro, ensure_ascii=False, default=str) + "\n"
        descriptor = os.open(
            os.fspath(ruta), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644
        )
        try:
            os.write(descriptor, linea.encode("utf-8"))
        finally:
            os.close(descriptor)
    except OSError as exc:
        _logger.warning("No se pudo escribir el evento de progreso %s: %s", evento, exc)


def _evento_fin(
    ruta: str | Path, etiqueta: EtiquetaUnidad, registro: RegistroUnidad
) -> None:
    emitir_evento(
        ruta,
        "unit_finished",
        **etiqueta.como_dict(),
        state=registro.estado,
        pid=registro.pid,
        started_at=_iso(registro.inicio),
        finished_at=_iso(registro.fin),
        seconds=round(registro.duracion_s, 3),
        cpu_s=None if registro.cpu_s is None else round(registro.cpu_s, 3),
        error=registro.error,
        code=registro.codigo,
    )


def _ejecutar_aislada[T, R](
    funcion: Callable[[T], R],
    tarea: T,
    etiqueta: EtiquetaUnidad | None = None,
    ruta: str | Path | None = None,
    etapa: str | None = None,
) -> ResultadoUnidad[R]:
    """Unidad de trabajo de nivel de modulo (``pickle`` no serializa closures).

    Con ``etiqueta`` y ``ruta`` anota el inicio y el fin de la unidad desde el
    propio proceso que la corre, en el instante en que ocurren.
    """
    seguimiento = etiqueta is not None and ruta is not None
    if seguimiento:
        assert etiqueta is not None and ruta is not None
        emitir_evento(
            ruta,
            "unit_started",
            **etiqueta.como_dict(),
            pid=os.getpid(),
            started_at=_iso(time.time()),
        )
    resultado = _correr_aislada(funcion, tarea, etapa)
    if seguimiento:
        assert etiqueta is not None and ruta is not None
        _evento_fin(ruta, etiqueta, resultado.registro)
    return resultado


def _correr_aislada[T, R](
    funcion: Callable[[T], R], tarea: T, etapa: str | None = None
) -> ResultadoUnidad[R]:
    inicio, cpu = time.time(), time.process_time()
    try:
        valor = funcion(tarea)
    except Exception as exc:
        registro = RegistroUnidad(
            "fallida",
            os.getpid(),
            inicio,
            time.time(),
            describir_error(exc),
            time.process_time() - cpu,
            codigo_de(exc, etapa=etapa),
        )
        return ResultadoUnidad(registro)
    registro = RegistroUnidad(
        "completada", os.getpid(), inicio, time.time(), cpu_s=time.process_time() - cpu
    )
    return ResultadoUnidad(registro, valor)


def describir_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def ejecutar_unidades[T, R](
    funcion: Callable[[T], R],
    tareas: Sequence[T],
    *,
    procesos: int = 1,
    etiquetas: Sequence[EtiquetaUnidad] | None = None,
    progreso: str | Path | None = None,
) -> list[ResultadoUnidad[R]]:
    """Ejecuta ``funcion(tarea)`` por tarea y devuelve los resultados en su orden.

    Una excepcion dentro de una unidad se registra como unidad ``fallida`` y no
    interrumpe a las demas. Si un proceso hijo muere (por ejemplo, por falta de
    memoria), el pool queda inservible: las unidades que no alcanzaron a
    terminar se registran como fallidas con esa causa, y una nueva ejecucion
    puede retomarlas.

    Con ``progreso`` (ruta de un ``.jsonl``) y ``etiquetas`` (una por tarea) cada
    unidad anota ``unit_started`` y ``unit_finished`` mientras corre; sin ellos el
    comportamiento no cambia.
    """
    if type(procesos) is not int or procesos < 1:
        raise ValueError(f"procesos debe ser un entero >= 1; se recibio {procesos!r}")
    if etiquetas is not None and len(etiquetas) != len(tareas):
        raise ValueError("etiquetas debe tener una entrada por tarea")
    seguimiento = etiquetas is not None and progreso is not None

    def etiqueta_de(indice: int) -> EtiquetaUnidad | None:
        return etiquetas[indice] if seguimiento and etiquetas is not None else None

    def etapa_de(indice: int) -> str | None:
        return etiquetas[indice].etapa if etiquetas is not None else None

    ruta = progreso if seguimiento else None
    if procesos == 1 or len(tareas) < 2:
        return [
            _ejecutar_aislada(
                funcion, tarea, etiqueta_de(indice), ruta, etapa_de(indice)
            )
            for indice, tarea in enumerate(tareas)
        ]

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
            ejecutor.submit(
                _ejecutar_aislada,
                funcion,
                tarea,
                etiqueta_de(indice),
                ruta,
                etapa_de(indice),
            ): indice
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
                        "fallida",
                        0,
                        envio,
                        time.time(),
                        describir_error(exc),
                        codigo=codigo_de(exc, etapa=etapa_de(indice)),
                    )
                    resultados[indice] = ResultadoUnidad(registro)
                    etiqueta = etiqueta_de(indice)
                    if etiqueta is not None and ruta is not None:
                        # El hijo murio sin poder anotar su fin: lo anota el padre.
                        _evento_fin(ruta, etiqueta, registro)
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
