"""Particion causal UNICA de Walk-Forward Validation.

La usan tanto el modo integro (componente 8.1, `walk_forward.py`) como el
modo greedy (componente 8.2, `walk_forward_greedy.py`). Si cada modo
generara su propio recorrido de ventanas, podrian diferir y producir
metricas no comparables entre si -- exactamente el riesgo que este archivo
existe para eliminar (ver docs/adr/ADR-003).
"""

from __future__ import annotations

from pred_engine.comun.dataclasses.validacion_temporal import VentanaTemporal
from pred_engine.comun.walkforward.errores import VentanaInsuficienteError


def generar_ventanas(
    n_observaciones: int, min_train: int, horizonte: int = 1, paso: int = 1
) -> list[VentanaTemporal]:
    if min_train < 1:
        raise ValueError("min_train debe ser >= 1")
    if horizonte < 1:
        raise ValueError("horizonte debe ser >= 1")
    if paso < 1:
        raise ValueError("paso debe ser >= 1")

    minimo_requerido = min_train + horizonte
    if n_observaciones < minimo_requerido:
        raise VentanaInsuficienteError(
            requerido=minimo_requerido, disponible=n_observaciones
        )

    ventanas: list[VentanaTemporal] = []
    fin_train = min_train
    while fin_train + horizonte <= n_observaciones:
        inicio_train = 0

        ventanas.append(
            VentanaTemporal(
                indice=len(ventanas),
                inicio_train=inicio_train,
                fin_train=fin_train,
                inicio_val=fin_train,
                fin_val=fin_train + horizonte,
            )
        )
        fin_train += paso

    return ventanas
