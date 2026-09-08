"""Componente 8.1: Walk-Forward Validation, modo completo.

Recorre TODAS las ventanas de una configuracion y devuelve las metricas
finales.

`tolerar_fallos=False` por defecto: este modo esta pensado como reporte o
auditoria final de una configuracion ya elegida (a diferencia del modo
greedy de `walk_forward_greedy.py`, que es el motor de BUSQUEDA que usa el
HPO), asi que un fallo de ajuste debe propagarse en vez de disolverse en un
`inf` -- quien pide un reporte final quiere saber si el modelo no ajusto,
no un numero que lo esconda. Contrastar con el modo greedy
(`tolerar_fallos=True` por defecto): ahi una ventana que no converge es
señal que ASHA puede usar para podar, no debe abortar el trial completo.
Ambos modos comparten `generar_ventanas`/`ejecutar_ventana` bit a bit (ver
docs/adr/ADR-003); la unica diferencia de comportamiento entre ellos es
esta, y esta deliberadamente documentada porque la prueba de equivalencia
(`tests/comun/walkforward/test_walk_forward_greedy.py`) usa un doble que
nunca falla y por lo tanto no la ejercita.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from pred_engine.comun.dataclasses.validacion_temporal import (
    ResultadoWalkForward,
)
from pred_engine.comun.logger import get_logger
from pred_engine.comun.walkforward._ejecucion import ejecutar_ventana
from pred_engine.comun.walkforward.metricas import valor_agregado_de
from pred_engine.comun.walkforward.protocolos import FabricaPronosticador
from pred_engine.comun.walkforward.ventanas import generar_ventanas

_logger = get_logger(__name__)


def evaluar_walk_forward(
    y: np.ndarray,
    fabrica: FabricaPronosticador,
    configuracion: Mapping[str, Any],
    min_train: int,
    horizonte: int = 1,
    paso: int = 1,
    metrica_objetivo: str = "mase",
    estacionalidad: int = 7,
    agregacion: Literal["media", "mediana", "media_recortada"] = "media_recortada",
    proporcion_recorte: float = 0.1,
    seed: int = 0,
    tolerar_fallos: bool = False,
    identificador: str = "",
) -> ResultadoWalkForward:
    serie = np.asarray(y, dtype=float)
    if serie.ndim != 1:
        raise ValueError("y debe ser un array 1D")
    else:
        ventanas_resueltas = generar_ventanas(
            len(serie), min_train=min_train, horizonte=horizonte, paso=paso
        )

    resultados = tuple(
        ejecutar_ventana(
            serie,
            fabrica,
            configuracion,
            ventana,
            estacionalidad=estacionalidad,
            seed=seed,
            identificador=identificador,
            tolerar_fallos=tolerar_fallos,
        )
        for ventana in ventanas_resueltas
    )

    valor_agregado = valor_agregado_de(
        resultados,
        metrica_objetivo,
        estrategia=agregacion,
        proporcion_recorte=proporcion_recorte,
    )

    _logger.info(
        "Walk-forward integro: %d/%d ventanas evaluadas | %s=%s",
        len(resultados),
        len(ventanas_resueltas),
        metrica_objetivo,
        valor_agregado,
    )

    return ResultadoWalkForward(
        ventanas=resultados,
        metrica_objetivo=metrica_objetivo,
        valor_agregado=valor_agregado,
        completo=True,
        n_ventanas_evaluadas=len(resultados),
        n_ventanas_totales=len(ventanas_resueltas),
    )
