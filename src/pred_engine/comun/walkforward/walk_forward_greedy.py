"""Componente 8.2: Walk-Forward Validation, modo greedy (paso a paso).

Reutiliza EXACTAMENTE la misma particion causal (`generar_ventanas`) y la
misma ejecucion por ventana (`_ejecucion.ejecutar_ventana`) que el modo
integro (`walk_forward.py`). Esto es deliberado: si este modulo
reimplementara su propio recorrido de ventanas, podria diferir del
recorrido "oficial" del modo integro y producir metricas no comparables
entre ambos modos (ver docs/adr/ADR-003).

Contrato central: `iterar_walk_forward` y `EjecutorGreedy` NO DECIDEN si se
poda. Solo ejecutan la siguiente ventana, actualizan el agregado parcial y
lo entregan. La decision de podar es responsabilidad EXCLUSIVA del
componente 5 (`optimizacion.optimizadores.HPO.asha.DecisorASHA`).

`tolerar_fallos=True` por defecto (a diferencia de `evaluar_walk_forward`,
que por defecto es `False`): este es el motor de BUSQUEDA que usa el HPO, y
una ventana que no converge es señal (para ASHA/poda semantica), no un
motivo para abortar el trial completo. Ver el docstring de
`walk_forward.py` para el contraste completo.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from pred_engine.comun.dataclasses.validacion_temporal import (
    EstadoParcial,
    ResultadoVentana,
    ResultadoWalkForward,
    VentanaTemporal,
)
from pred_engine.comun.logger import get_logger
from pred_engine.comun.walkforward._ejecucion import ejecutar_ventana
from pred_engine.comun.walkforward.metricas import valor_agregado_de
from pred_engine.comun.walkforward.protocolos import FabricaPronosticador

_logger = get_logger(__name__)


def iterar_walk_forward(
    y: np.ndarray,
    fabrica: FabricaPronosticador,
    configuracion: Mapping[str, Any],
    *,
    ventanas: Sequence[VentanaTemporal],
    metrica_objetivo: str = "mase",
    estacionalidad: int = 7,
    agregacion: Literal["media", "mediana", "media_recortada"] = "media_recortada",
    proporcion_recorte: float = 0.1,
    seed: int = 0,
    tolerar_fallos: bool = True,
    identificador: str = "",
) -> Generator[EstadoParcial, None, None]:
    serie = np.asarray(y, dtype=float)
    if serie.ndim != 1:
        raise ValueError("y debe ser un array 1D")
    if not ventanas:
        raise ValueError("ventanas no puede ser vacio en el modo greedy")

    historial: list[ResultadoVentana] = []
    for ventana in ventanas:
        resultado = ejecutar_ventana(
            serie,
            fabrica,
            configuracion,
            ventana,
            estacionalidad=estacionalidad,
            seed=seed,
            identificador=identificador,
            tolerar_fallos=tolerar_fallos,
        )
        historial.append(resultado)
        valor_parcial = valor_agregado_de(
            historial,
            metrica_objetivo,
            estrategia=agregacion,
            proporcion_recorte=proporcion_recorte,
        )
        yield EstadoParcial(
            ultima=resultado,
            n_evaluadas=len(historial),
            n_totales=len(ventanas),
            valor_parcial=valor_parcial,
        )


class EjecutorGreedy:
    def __init__(
        self,
        y: np.ndarray,
        fabrica: FabricaPronosticador,
        configuracion: Mapping[str, Any],
        *,
        ventanas: Sequence[VentanaTemporal],
        metrica_objetivo: str = "mase",
        estacionalidad: int = 7,
        agregacion: Literal["media", "mediana", "media_recortada"] = "media_recortada",
        proporcion_recorte: float = 0.1,
        seed: int = 0,
        tolerar_fallos: bool = True,
        identificador: str = "",
    ) -> None:
        self._ventanas = tuple(ventanas)
        self._metrica_objetivo = metrica_objetivo
        self._agregacion: Literal["media", "mediana", "media_recortada"] = agregacion
        self._proporcion_recorte = proporcion_recorte
        self._generador = iterar_walk_forward(
            y,
            fabrica,
            configuracion,
            ventanas=self._ventanas,
            metrica_objetivo=metrica_objetivo,
            estacionalidad=estacionalidad,
            agregacion=agregacion,
            proporcion_recorte=proporcion_recorte,
            seed=seed,
            tolerar_fallos=tolerar_fallos,
            identificador=identificador,
        )
        self._historial: list[ResultadoVentana] = []
        self._cerrado = False
        self._motivo_cierre: str | None = None

    @property
    def motivo_cierre(self) -> str | None:
        return self._motivo_cierre

    def avanzar(self) -> EstadoParcial | None:
        if self._cerrado:
            return None
        try:
            estado = next(self._generador)
        except StopIteration:
            self._cerrado = True
            return None
        self._historial.append(estado.ultima)
        return estado

    def cerrar(self, motivo: str) -> None:
        if self._cerrado:
            return
        self._motivo_cierre = motivo
        self._generador.close()
        self._cerrado = True
        _logger.info(
            "EjecutorGreedy cerrado | motivo=%s | ventanas=%d/%d",
            motivo,
            len(self._historial),
            len(self._ventanas),
        )

    def resultado(self) -> ResultadoWalkForward:
        valor_agregado = valor_agregado_de(
            self._historial,
            self._metrica_objetivo,
            estrategia=self._agregacion,
            proporcion_recorte=self._proporcion_recorte,
        )
        completo = self._motivo_cierre is None and len(self._historial) == len(
            self._ventanas
        )
        return ResultadoWalkForward(
            ventanas=tuple(self._historial),
            metrica_objetivo=self._metrica_objetivo,
            valor_agregado=valor_agregado,
            completo=completo,
            n_ventanas_evaluadas=len(self._historial),
            n_ventanas_totales=len(self._ventanas),
        )
