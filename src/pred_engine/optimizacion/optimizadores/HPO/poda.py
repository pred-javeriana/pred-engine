"""Regla #3 (poda semantica): descarta configuraciones que enganan al sistema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# Prefijo del motivo de un trial descartado por la regla #3; ASHA lo usa para no
# comparar contra el valor parcial de una configuracion degenerada.
PREFIJO_PODA_SEMANTICA = "poda_semantica"


@dataclass(frozen=True, slots=True)
class ReglasPoda:
    min_ventanas: int = 4
    agregacion: Literal["media", "mediana", "media_recortada"] = "media_recortada"
    proporcion_recorte: float = 0.1
    factor_reduccion: int = 3
    habilitar_poda_semantica: bool = True


def es_degenerada(
    y_pred: np.ndarray, *, y_train: np.ndarray, tol: float = 1e-9
) -> tuple[bool, str | None]:
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if not np.all(np.isfinite(y_pred)):
        return True, "prediccion_no_finita"
    if np.any(y_pred < -tol):
        return True, "demanda_negativa"
    if np.all(y_pred <= tol) and np.any(y_train > tol):
        return True, "prediccion_nula"
    return False, None
