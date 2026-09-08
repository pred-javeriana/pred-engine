"""Calculo sin estado del cuadrado del coeficiente de variacion (CV²)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from pred_engine.comun.logger import get_logger
from pred_engine.ingesta.categorizacion.errores import TopologyMathError

_logger = get_logger(__name__)


def compute_cv2(demand: ArrayLike) -> float:
    """CV² sobre el subconjunto con demanda estrictamente positiva.

    Excluye ceros, no finitos y negativos. Si solo hay una observacion
    positiva, no hay variacion muestral: se devuelve 0.0.
    """
    serie = np.asarray(demand, dtype=np.float64)
    if serie.ndim != 1:
        raise TopologyMathError("la demanda para CV2 debe ser un arreglo 1-D")
    positivos = serie[np.isfinite(serie) & (serie > 0.0)]
    if positivos.size == 0:
        _logger.error("CV2 indefinido: no hay demandas estrictamente positivas")
        raise TopologyMathError(
            "no hay demanda estrictamente positiva para calcular CV2"
        )
    if positivos.size == 1:
        _logger.info("CV2=0.0: una sola demanda positiva (sin variacion)")
        return 0.0
    media = float(np.mean(positivos))
    # media > 0 porque el filtro es estricto; se guarda igualmente el fail-fast.
    if media == 0.0:
        raise TopologyMathError("division por cero: media de demanda positiva nula")
    sigma = float(np.std(positivos, ddof=1))
    cv2 = float((sigma / media) ** 2)
    _logger.info(
        "CV2 calculado n_pos=%s media=%.6f cv2=%.6f",
        positivos.size,
        media,
        cv2,
    )
    return cv2
