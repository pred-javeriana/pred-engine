"""Diebold-Mariano con correccion HLN y multiplicidad BH entre SKUs.

Implementa la alternativa 2 del ADR-03-008: el campeon de cada SKU contra
Seasonal Naive, solo con datos reales. La perdida es el error cuadratico medio
de cada ventana, coherente con r = RMSE_c / RMSE_SN (ADR-03-007). Con ventanas
de horizonte H que se solapan, el diferencial se autocorrelaciona hasta el
rezago H - 1: la varianza suma esas autocovarianzas, como `forecast::dm.test`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from pred_engine.comun.logger import get_logger
from pred_engine.forecasting.evaluaciones.calculo_errores import MetricasCandidato

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PruebaDM:
    """HLN-DM del campeon contra Seasonal Naive; `causa` si no se calculo.

    `estadistico` < 0 significa que el campeon tiene menor perdida. El p-valor
    es bilateral; `p_ajustado`, `significativa` y su nivel `alfa` llegan con la
    correccion BH.
    """

    n_ventanas: int
    horizonte: int
    estadistico: float | None = None
    p_valor: float | None = None
    p_ajustado: float | None = None
    significativa: bool | None = None
    alfa: float | None = None
    causa: str | None = None


def prueba_hln(diferencias: np.ndarray, horizonte: int) -> PruebaDM:
    """DM con la correccion de Harvey, Leybourne y Newbold (1997), t con T-1 gl."""
    if horizonte < 1:
        raise ValueError("el horizonte debe ser >= 1")
    d = np.asarray(diferencias, dtype=float)
    t = len(d)
    if t < 2 or t <= horizonte:
        return PruebaDM(t, horizonte, causa="pocas_ventanas")
    centradas = d - d.mean()
    autocovarianzas = [
        float(np.dot(centradas[k:], centradas[: t - k])) / t for k in range(horizonte)
    ]
    varianza = (autocovarianzas[0] + 2 * sum(autocovarianzas[1:])) / t
    if not varianza > 0:
        return PruebaDM(t, horizonte, causa="varianza_no_positiva")
    h = horizonte
    estadistico = (
        float(d.mean())
        / math.sqrt(varianza)
        * math.sqrt((t + 1 - 2 * h + h * (h - 1) / t) / t)
    )
    p_valor = float(2 * stats.t.sf(abs(estadistico), df=t - 1))
    return PruebaDM(t, horizonte, estadistico=estadistico, p_valor=p_valor)


def probar_contra_linea_base(
    campeon: MetricasCandidato, linea_base: MetricasCandidato
) -> PruebaDM:
    """HLN-DM sobre las ventanas validas para el campeon y para Seasonal Naive."""
    base = {v.origen: v.metricas for v in linea_base.por_ventana}
    pares = [
        (v.metricas, base[v.origen]) for v in campeon.por_ventana if v.origen in base
    ]
    if not pares:
        return PruebaDM(0, 0, causa="sin_ventanas_comunes")
    diferencias = np.array([c.rmse**2 - b.rmse**2 for c, b in pares])
    return prueba_hln(diferencias, max(c.n_pares for c, _ in pares))


def corregir_multiplicidad(
    pruebas: Mapping[str, PruebaDM], alfa: float
) -> dict[str, PruebaDM]:
    """Benjamini-Hochberg sobre los SKUs con p-valor; los demas quedan igual."""
    calculadas = [sku for sku, prueba in pruebas.items() if prueba.p_valor is not None]
    corregidas = dict(pruebas)
    if calculadas:
        rechazos, ajustados, _, _ = multipletests(
            [pruebas[sku].p_valor for sku in calculadas], alpha=alfa, method="fdr_bh"
        )
        for sku, rechazo, ajustado in zip(calculadas, rechazos, ajustados, strict=True):
            corregidas[sku] = replace(
                pruebas[sku],
                p_ajustado=float(ajustado),
                significativa=bool(rechazo),
                alfa=alfa,
            )
    _logger.info(
        "Diebold-Mariano HLN: %d pruebas, %d significativas (BH, alfa=%s)",
        len(calculadas),
        sum(1 for p in corregidas.values() if p.significativa),
        alfa,
    )
    return corregidas
