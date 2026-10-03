"""Muestreo de CPU y memoria de una corrida, por proceso, mientras corre.

Una hebra del proceso principal toma una muestra cada ``interval`` segundos del
proceso principal, de cada proceso hijo (los procesos del pool de L2-L3) y de la
maquina, y la agrega como lineas JSON a ``recursos.jsonl``. Cada linea se escribe
y se vacia al momento: si la corrida muere a mitad de camino (por ejemplo, por
falta de memoria), las muestras previas quedan en disco.

Formato de cada linea: ``at`` (ISO 8601, UTC), ``role``, ``pid``, ``cpu_cores``
y ``mem_mb``.

- ``host``: una sola linea inicial con la capacidad de la maquina (nucleos
  logicos y memoria total).
- ``system``: la maquina entera (nucleos ocupados y memoria en uso, es decir,
  total menos disponible).
- ``main``, ``worker`` y ``child``: el proceso principal, los procesos del pool
  (``spawn``) y cualquier otro descendiente. ``cpu_cores`` es el promedio de
  nucleos usados desde la muestra anterior (1.0 = un nucleo completo) y
  ``mem_mb`` la memoria residente (RSS). El RSS de varios procesos cuenta dos
  veces las paginas compartidas; ``system`` no.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

import psutil

from pred_engine.comun.logger import get_logger

_logger = get_logger(__name__)
_MB = 1024 * 1024


def _instante(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).isoformat()


def _rol(proceso: psutil.Process) -> str:
    try:
        linea = " ".join(proceso.cmdline())
    except psutil.Error:
        return "child"
    return "worker" if "spawn_main" in linea else "child"


class ResourceSampler:
    """Hebra de muestreo; usar como contexto alrededor de la corrida.

    El costo propio de la hebra (CPU y milisegundos por muestra) queda en
    ``summary()`` para que la corrida declare cuanto le costo medirse.
    """

    def __init__(self, path: str | Path, interval: float = 1.0) -> None:
        if not interval > 0:
            raise ValueError("interval must be positive")
        self.path = Path(path)
        self.interval = interval
        self._raiz = psutil.Process()
        self._procesos: dict[int, psutil.Process] = {}
        self._roles: dict[int, str] = {}
        self._previo: dict[int, tuple[float, float]] = {}
        self._alto = threading.Event()
        self._hebra: threading.Thread | None = None
        self._archivo: IO[str] | None = None
        self._muestras = 0
        self._cpu_propio = 0.0
        self._reloj_propio = 0.0
        self._pico_cpu = 0.0
        self._pico_mem = 0.0
        self._pico_workers = 0
        self._fallo: str | None = None

    def __enter__(self) -> ResourceSampler:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._hebra is not None:
            raise RuntimeError("sampler already started")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._archivo = self.path.open("w", encoding="utf-8")
        ahora = time.time()
        memoria = psutil.virtual_memory()
        self._escribir(
            [
                {
                    "at": _instante(ahora),
                    "role": "host",
                    "pid": None,
                    "cpu_cores": psutil.cpu_count() or 1,
                    "mem_mb": round(memoria.total / _MB, 1),
                }
            ]
        )
        psutil.cpu_percent(interval=None)
        tiempos = self._raiz.cpu_times()
        self._procesos[self._raiz.pid] = self._raiz
        self._roles[self._raiz.pid] = "main"
        self._previo[self._raiz.pid] = (ahora, tiempos.user + tiempos.system)
        self._hebra = threading.Thread(
            target=self._bucle, name="pred-resource-sampler", daemon=True
        )
        self._hebra.start()

    def stop(self) -> None:
        if self._hebra is None or self._archivo is None:
            return
        self._alto.set()
        self._hebra.join()
        self._muestrear()
        self._archivo.close()
        self._archivo = None

    def _bucle(self) -> None:
        while not self._alto.wait(self.interval):
            self._muestrear()

    def _muestrear(self) -> None:
        """Una muestra; un error nunca interrumpe la corrida, solo el muestreo."""
        if self._fallo is not None:
            return
        cpu_inicio, reloj_inicio = time.thread_time(), time.perf_counter()
        try:
            self._escribir(self._filas())
        except Exception as exc:
            self._fallo = f"{type(exc).__name__}: {exc}"
            _logger.error("Muestreo de recursos detenido: %s", self._fallo)
        self._muestras += 1
        self._cpu_propio += time.thread_time() - cpu_inicio
        self._reloj_propio += time.perf_counter() - reloj_inicio

    def _filas(self) -> list[dict[str, Any]]:
        try:
            hijos = self._raiz.children(recursive=True)
        except psutil.Error:
            hijos = []
        vivos = {self._raiz.pid: self._raiz}
        for hijo in hijos:
            vivos[hijo.pid] = self._procesos.get(hijo.pid, hijo)
        self._procesos = vivos
        ahora = time.time()
        memoria = psutil.virtual_memory()
        ocupados = psutil.cpu_percent(interval=None) / 100 * (psutil.cpu_count() or 1)
        filas: list[dict[str, Any]] = [
            {
                "at": _instante(ahora),
                "role": "system",
                "pid": None,
                "cpu_cores": round(ocupados, 2),
                "mem_mb": round((memoria.total - memoria.available) / _MB, 1),
            }
        ]
        cpu_total = mem_total = 0.0
        workers = 0
        for pid, proceso in vivos.items():
            try:
                with proceso.oneshot():
                    tiempos = proceso.cpu_times()
                    rss = proceso.memory_info().rss
                    if pid not in self._previo:
                        self._previo[pid] = (proceso.create_time(), 0.0)
                    if pid not in self._roles:
                        self._roles[pid] = _rol(proceso)
            except psutil.Error:
                continue
            usado = tiempos.user + tiempos.system
            antes, usado_antes = self._previo[pid]
            self._previo[pid] = (ahora, usado)
            cpu = max(0.0, (usado - usado_antes) / max(ahora - antes, 1e-6))
            cpu_total += cpu
            mem_total += rss / _MB
            workers += self._roles[pid] == "worker"
            filas.append(
                {
                    "at": _instante(ahora),
                    "role": self._roles[pid],
                    "pid": pid,
                    "cpu_cores": round(cpu, 3),
                    "mem_mb": round(rss / _MB, 1),
                }
            )
        self._pico_cpu = max(self._pico_cpu, cpu_total)
        self._pico_mem = max(self._pico_mem, mem_total)
        self._pico_workers = max(self._pico_workers, workers)
        return filas

    def _escribir(self, filas: list[dict[str, Any]]) -> None:
        assert self._archivo is not None
        self._archivo.write(
            "".join(json.dumps(fila, ensure_ascii=False) + "\n" for fila in filas)
        )
        self._archivo.flush()

    def summary(self) -> dict[str, Any]:
        """Resumen para ``corrida.json``: picos y costo propio del muestreo."""
        return {
            "file": self.path.name,
            "interval_s": self.interval,
            "samples": self._muestras,
            "processes": len(self._roles),
            "peak_cpu_cores": round(self._pico_cpu, 2),
            "peak_rss_mb": round(self._pico_mem, 1),
            "peak_workers": self._pico_workers,
            "sampler_cpu_s": round(self._cpu_propio, 3),
            "sampler_ms_per_sample": (
                round(1000 * self._reloj_propio / self._muestras, 2)
                if self._muestras
                else None
            ),
            "error": self._fallo,
        }
