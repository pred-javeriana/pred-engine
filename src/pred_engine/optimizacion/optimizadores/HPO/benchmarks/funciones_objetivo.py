"""Funciones objetivo sinteticas con minimo global conocido (0.0 en todas).

Estandar de la literatura de HPO/optimizacion global (ver Bergstra et al.,
TPE; Li et al., ASHA) para comparar samplers sin el costo/ruido de un ajuste
real. Cada funcion recibe la configuracion sugerida por un `optuna.trial`
(claves `"x0", "x1", ...`) y no conoce Optuna ni el motor de HPO -- son
funciones puras de `Mapping[str, float] -> float`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _vector(configuracion: Mapping[str, float]) -> np.ndarray:
    claves = sorted(configuracion, key=lambda clave: int(clave[1:]))
    return np.array([configuracion[clave] for clave in claves], dtype=float)


def sphere(configuracion: Mapping[str, float]) -> float:
    """Convexa, separable: el caso mas facil. Minimo 0.0 en el origen."""
    x = _vector(configuracion)
    return float(np.sum(x**2))


def rastrigin(configuracion: Mapping[str, float]) -> float:
    """Altamente multimodal (muchos minimos locales regularmente
    espaciados) pero separable. Minimo 0.0 en el origen."""
    x = _vector(configuracion)
    return float(10.0 * len(x) + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def rosenbrock(configuracion: Mapping[str, float]) -> float:
    """No separable: un valle curvo estrecho ('banana'). Minimo 0.0 en
    (1, 1, ..., 1). Requiere dim >= 2."""
    x = _vector(configuracion)
    if x.size < 2:
        raise ValueError("rosenbrock requiere al menos 2 dimensiones")
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))
