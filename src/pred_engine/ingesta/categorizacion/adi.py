"""Calculo sin estado del Intervalo Promedio de Demanda (ADI)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from pred_engine.comun.logger import get_logger
from pred_engine.ingesta.categorizacion.errores import TopologyMathError

_logger = get_logger(__name__)


def compute_adi(demand: ArrayLike) -> float:
    """ADI = n / n(d>0) sobre observaciones finitas.

    Ignora NaN al definir periodos evaluados. Ignora nulos y valores
    <= 0 al contar demanda activa. No muta el arreglo de entrada.
    """
    serie = np.asarray(demand, dtype=np.float64)
    if serie.ndim != 1:
        raise TopologyMathError("la demanda para ADI debe ser un arreglo 1-D")
    evaluados = serie[np.isfinite(serie)]
    if evaluados.size == 0:
        raise TopologyMathError("no hay periodos evaluables para ADI")
    n_activos = int(np.sum(evaluados > 0.0))
    if n_activos == 0:
        _logger.error("ADI indefinido: cero demandas estrictamente positivas")
        raise TopologyMathError(
            "division por cero: ninguna demanda estrictamente positiva"
        )
    adi = float(evaluados.size / n_activos)
    _logger.info("ADI calculado n=%s n_pos=%s adi=%.6f", evaluados.size, n_activos, adi)
    return adi
