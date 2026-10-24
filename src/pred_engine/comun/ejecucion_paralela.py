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
import threading
import time
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
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


def iniciar_proceso_del_pool() -> None:
    """`initializer` de cada proceso del pool: BLAS de una hebra y fin con el padre.

    Si el padre muere sin cerrar el pool (por ejemplo, lo mata el sistema por
    falta de memoria), sus hijos quedarian esperando tareas para siempre con su
    memoria tomada. Una hebra del hijo nota el cambio de padre y lo termina.
    """
    fijar_una_hebra_blas()
    padre = os.getppid()

    def vigilar_padre() -> None:
        while os.getppid() == padre:
            time.sleep(1.0)
        os._exit(1)

    threading.Thread(target=vigilar_padre, name="vigilar-padre", daemon=True).start()


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
    _exigir_procesos(procesos)
    if etiquetas is not None and len(etiquetas) != len(tareas):
        raise ValueError("etiquetas debe tener una entrada por tarea")
    efectivos = max(1, min(procesos, len(tareas)))
    if efectivos > 1:
        _logger.info(
            "Ejecucion en paralelo | unidades=%d | procesos=%d", len(tareas), efectivos
        )
    return list(
        iterar_unidades(
            funcion,
            tareas,
            procesos=efectivos,
            en_vuelo=max(1, len(tareas)),
            etiquetas=etiquetas,
            progreso=progreso,
        )
    )


def iterar_unidades[T, R](
    funcion: Callable[[T], R],
    tareas: Iterable[T],
    *,
    procesos: int = 1,
    en_vuelo: int | None = None,
    etiquetas: Sequence[EtiquetaUnidad] | None = None,
    progreso: str | Path | None = None,
) -> Generator[ResultadoUnidad[R]]:
    """Como ``ejecutar_unidades``, pero entrega cada resultado apenas esta listo.

    Los resultados salen en el orden de las tareas. Las tareas se leen de a
    poco: a lo sumo ``en_vuelo`` (por defecto, dos por proceso) quedan enviadas
    sin entregar, asi que ni las entradas ni los resultados de todas las tareas
    tienen que caber a la vez en memoria. Quien consume puede, por ejemplo,
    guardar cada resultado antes de pedir el siguiente. Si deja de consumir,
    las tareas no iniciadas se cancelan.
    """
    _exigir_procesos(procesos)
    limite = 2 * procesos if en_vuelo is None else en_vuelo
    if type(limite) is not int or limite < 1:
        raise ValueError(f"en_vuelo debe ser un entero >= 1; se recibio {en_vuelo!r}")
    return _iterar(funcion, tareas, procesos, limite, etiquetas, progreso)


def _exigir_procesos(procesos: int) -> None:
    if type(procesos) is not int or procesos < 1:
        raise ValueError(f"procesos debe ser un entero >= 1; se recibio {procesos!r}")


def _iterar[T, R](
    funcion: Callable[[T], R],
    tareas: Iterable[T],
    procesos: int,
    limite: int,
    etiquetas: Sequence[EtiquetaUnidad] | None,
    ruta: str | Path | None,
) -> Generator[ResultadoUnidad[R]]:
    def etiqueta_de(indice: int) -> EtiquetaUnidad | None:
        return None if etiquetas is None else etiquetas[indice]

    def etapa_de(indice: int) -> str | None:
        return None if etiquetas is None else etiquetas[indice].etapa

    if procesos == 1:
        for indice, tarea in enumerate(tareas):
            yield _ejecutar_aislada(
                funcion, tarea, etiqueta_de(indice), ruta, etapa_de(indice)
            )
        return

    with (
        blas_de_una_hebra(),
        ProcessPoolExecutor(
            max_workers=procesos,
            mp_context=CONTEXTO_MP,
            initializer=iniciar_proceso_del_pool,
        ) as ejecutor,
    ):
        envio = time.time()
        enviadas: deque[tuple[int, Future[ResultadoUnidad[R]]]] = deque()
        pendientes = enumerate(tareas)
        try:
            while True:
                for indice, tarea in pendientes:
                    enviadas.append(
                        (
                            indice,
                            _enviar(
                                ejecutor,
                                funcion,
                                tarea,
                                etiqueta_de(indice),
                                ruta,
                                etapa_de(indice),
                            ),
                        )
                    )
                    if len(enviadas) >= limite:
                        break
                if not enviadas:
                    return
                indice, futuro = enviadas.popleft()
                try:
                    resultado = futuro.result()
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
                    resultado = ResultadoUnidad(registro)
                    etiqueta = etiqueta_de(indice)
                    if etiqueta is not None and ruta is not None:
                        # El hijo murio sin poder anotar su fin: lo anota el padre.
                        _evento_fin(ruta, etiqueta, registro)
                yield resultado
        except BaseException:
            # Una interrupcion no debe dejar corriendo las unidades no iniciadas.
            for _, pendiente in enviadas:
                pendiente.cancel()
            raise


def _enviar[T, R](
    ejecutor: ProcessPoolExecutor,
    funcion: Callable[[T], R],
    tarea: T,
    etiqueta: EtiquetaUnidad | None,
    ruta: str | Path | None,
    etapa: str | None = None,
) -> Future[ResultadoUnidad[R]]:
    try:
        return ejecutor.submit(_ejecutar_aislada, funcion, tarea, etiqueta, ruta, etapa)
    except BrokenProcessPool as exc:
        # Un hijo ya murio: la tarea queda fallida con esa causa, como las enviadas.
        futuro: Future[ResultadoUnidad[R]] = Future()
        futuro.set_exception(exc)
        return futuro


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
