"""Series sinteticas compartidas por las pruebas.

No empieza con `test_`: pytest no lo recolecta como modulo de pruebas, solo
se importa desde los `test_*.py` que lo necesitan.
"""

from __future__ import annotations

import numpy as np


def serie_ar_estacional(
    n: int,
    *,
    nivel: float = 30.0,
    phi: float = 0.5,
    phi_estacional: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """(1 - phi B)(1 - phi_estacional B^7)(y_t - nivel) = ruido N(0, 1).

    Estacionaria y con estructura semanal: los ajustes SARIMA quedan lejos del
    borde de la region admisible (ADR-020). Una serie determinista, en cambio,
    deja raices AR o MA estacionales en el circulo unitario y se rechaza.
    """
    rng = np.random.default_rng(seed)
    calentamiento = 20
    e = np.zeros(n + calentamiento)
    for t in range(8, n + calentamiento):
        e[t] = (
            phi * e[t - 1]
            + phi_estacional * e[t - 7]
            - phi * phi_estacional * e[t - 8]
            + rng.normal(0, 1)
        )
    return nivel + e[calentamiento:]
