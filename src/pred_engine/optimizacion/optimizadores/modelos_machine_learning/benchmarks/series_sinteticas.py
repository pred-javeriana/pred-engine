"""Series sinteticas reproducibles por clase de SKU (semilla fija, m=7)."""

from __future__ import annotations

from typing import Literal

import numpy as np

# Solo las clases que la politica 2.2 enruta a la familia ML.
ClaseML = Literal["smooth", "erratic", "intermittent"]
CLASES_ML: tuple[ClaseML, ...] = ("smooth", "erratic", "intermittent")


def serie_sintetica(clase: ClaseML, n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    if clase == "smooth":
        y = 20 + 6 * np.sin(2 * np.pi * t / 7) + 0.02 * t + rng.normal(0, 1.5, n)
    elif clase == "erratic":
        picos = 1 + 3 * (rng.random(n) < 0.15)
        y = 20 + 8 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1, n) * picos
    elif clase == "intermittent":
        con_demanda = rng.random(n) >= 0.55
        y = np.where(con_demanda, 10 + 6 * np.sin(2 * np.pi * t / 7), 0.0)
        y = y + np.where(con_demanda, rng.normal(0, 2, n), 0.0)
    else:
        raise ValueError(f"clase no soportada por la familia ML: {clase!r}")
    return np.maximum(0.0, y)
