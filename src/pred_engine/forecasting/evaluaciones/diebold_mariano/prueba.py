"""Diebold-Mariano con correccion HLN, intervalo de r y multiplicidad entre SKUs.

Implementa ADR-03-011: el campeon de cada SKU contra
Seasonal Naive, solo con datos reales. La perdida es el error cuadratico medio
de cada ventana, coherente con r = RMSE_c / RMSE_SN (ADR-03-007). Con ventanas
de horizonte H que se solapan, el diferencial se autocorrelaciona hasta el
rezago H - 1: la varianza suma esas autocovarianzas, como `forecast::dm.test`.
El intervalo de r invierte esa misma prueba (ADR-023).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from pred_engine.comun.logger import get_logger
from pred_engine.forecasting.evaluaciones.calculo_errores import (
    MetricasCandidato,
    pares_comunes,
)

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


@dataclass(frozen=True, slots=True)
class IntervaloRazon:
    """Valores de r que HLN-DM no rechaza al ``nivel`` dado (ADR-023).

    Contiene a r y excluye 1 justo cuando la prueba rechaza a ese nivel.
    ``superior`` es ``None`` cuando el intervalo no tiene cota superior: la
    perdida de Seasonal Naive no se distingue de cero a ese nivel.
    """

    nivel: float
    inferior: float
    superior: float | None


def prueba_hln(diferencias: np.ndarray, horizonte: int) -> PruebaDM:
    """DM con la correccion de Harvey, Leybourne y Newbold (1997), t con T-1 gl."""
    if horizonte < 1:
        raise ValueError("el horizonte debe ser >= 1")
    d = np.asarray(diferencias, dtype=float)
    t = len(d)
    if t < 2 or t <= horizonte:
        return PruebaDM(t, horizonte, causa="pocas_ventanas")
    varianza = _varianza_largo_plazo(d, horizonte) / t
    if not varianza > 0:
        return PruebaDM(t, horizonte, causa="varianza_no_positiva")
    estadistico = float(d.mean()) / math.sqrt(varianza) * _factor_hln(t, horizonte)
    p_valor = float(2 * stats.t.sf(abs(estadistico), df=t - 1))
    return PruebaDM(t, horizonte, estadistico=estadistico, p_valor=p_valor)


def probar_contra_linea_base(
    campeon: MetricasCandidato,
    linea_base: MetricasCandidato,
    *,
    n_min_ventanas: int = 2,
) -> PruebaDM:
    """HLN-DM sobre origenes comunes, con minimo predeclarado y horizonte H."""
    return probar_pares(campeon, linea_base, n_min_ventanas=n_min_ventanas)


def probar_pares(
    a: MetricasCandidato, b: MetricasCandidato, *, n_min_ventanas: int = 2
) -> PruebaDM:
    """HLN-DM de ``a`` contra ``b`` sobre sus ventanas comunes; < 0 favorece a ``a``."""
    if type(n_min_ventanas) is not int or n_min_ventanas < 2:
        raise ValueError("n_min_ventanas debe ser un entero >= 2")
    pares = pares_comunes(a, b)
    if not pares:
        return PruebaDM(0, 0, causa="sin_ventanas_comunes")
    horizonte = max(c.n_pares for c, _ in pares)
    if len(pares) < n_min_ventanas:
        return PruebaDM(len(pares), horizonte, causa="pocas_ventanas")
    diferencias = np.array([c.rmse**2 - d.rmse**2 for c, d in pares])
    return prueba_hln(diferencias, horizonte)


def intervalo_razon(
    campeon: MetricasCandidato, linea_base: MetricasCandidato, *, nivel: float
) -> IntervaloRazon | None:
    """Invierte HLN-DM sobre las ventanas comunes: CI(r) = {rho : |HLN| <= t}.

    Con perdidas a (campeon) y b (linea base), el diferencial a - u * b con
    u = rho ** 2 tiene media y varianza de largo plazo cuadraticas en u, asi que
    la region aceptada es la de una cuadratica (Fieller, 1954). ``None`` si la
    prueba no se puede calcular con esas ventanas.
    """
    if not 0 < nivel < 1:
        raise ValueError("nivel debe estar en (0, 1)")
    pares = pares_comunes(campeon, linea_base)
    if not pares:
        return None
    horizonte = max(c.n_pares for c, _ in pares)
    a = np.array([c.rmse**2 for c, _ in pares])
    b = np.array([d.rmse**2 for _, d in pares])
    t = len(a)
    if t < 2 or t <= horizonte:
        return None
    critico = float(stats.t.ppf(1 - (1 - nivel) / 2, df=t - 1))
    kappa = critico**2 / (t * _factor_hln(t, horizonte) ** 2)
    var_a = _varianza_largo_plazo(a, horizonte)
    var_b = _varianza_largo_plazo(b, horizonte)
    cruzada = (var_a + var_b - _varianza_largo_plazo(a - b, horizonte)) / 2
    media_a, media_b = float(a.mean()), float(b.mean())
    aceptados = _no_positivos(
        media_b**2 - kappa * var_b,
        media_a * media_b - kappa * cruzada,
        media_a**2 - kappa * var_a,
    )
    if aceptados is None:
        return None
    inferior, superior = aceptados
    return IntervaloRazon(
        nivel,
        math.sqrt(inferior),
        None if math.isinf(superior) else math.sqrt(superior),
    )


def corregir_multiplicidad(
    pruebas: Mapping[str, PruebaDM],
    alfa: float,
    *,
    metodo: Literal["fdr_bh", "fdr_by"] = "fdr_bh",
) -> dict[str, PruebaDM]:
    """BH (o BY) sobre los SKUs con p-valor; los demas quedan igual."""
    calculadas = [sku for sku, prueba in pruebas.items() if prueba.p_valor is not None]
    corregidas = dict(pruebas)
    if calculadas:
        rechazos, ajustados, _, _ = multipletests(
            [pruebas[sku].p_valor for sku in calculadas], alpha=alfa, method=metodo
        )
        for sku, rechazo, ajustado in zip(calculadas, rechazos, ajustados, strict=True):
            corregidas[sku] = replace(
                pruebas[sku],
                p_ajustado=float(ajustado),
                significativa=bool(rechazo),
                alfa=alfa,
            )
    _logger.info(
        "Diebold-Mariano HLN: %d pruebas, %d significativas (%s, alfa=%s)",
        len(calculadas),
        sum(1 for p in corregidas.values() if p.significativa),
        metodo,
        alfa,
    )
    return corregidas


def _varianza_largo_plazo(d: np.ndarray, horizonte: int) -> float:
    """gamma_0 + 2 * sum(gamma_k), k = 1..H-1: nucleo rectangular hasta H - 1."""
    t = len(d)
    centradas = d - d.mean()
    autocovarianzas = [
        float(np.dot(centradas[k:], centradas[: t - k])) / t for k in range(horizonte)
    ]
    return autocovarianzas[0] + 2 * sum(autocovarianzas[1:])


def _factor_hln(t: int, h: int) -> float:
    return math.sqrt((t + 1 - 2 * h + h * (h - 1) / t) / t)


def _no_positivos(alfa: float, beta: float, gamma: float) -> tuple[float, float] | None:
    """[inf, sup] de {u >= 0 : alfa u^2 - 2 beta u + gamma <= 0}; sup puede ser inf."""
    if alfa == 0:
        if beta == 0:
            return (0.0, math.inf) if gamma <= 0 else None
        raiz = gamma / (2 * beta)
        if beta > 0:
            return max(raiz, 0.0), math.inf
        return (0.0, raiz) if raiz >= 0 else None
    discriminante = beta * beta - alfa * gamma
    if alfa > 0:
        if discriminante < 0:
            return None
        raiz = math.sqrt(discriminante)
        menor, mayor = (beta - raiz) / alfa, (beta + raiz) / alfa
        return None if mayor < 0 else (max(menor, 0.0), mayor)
    # alfa < 0: parabola abierta hacia abajo, se acepta fuera de las raices.
    if discriminante <= 0:
        return 0.0, math.inf
    raiz = math.sqrt(discriminante)
    menor, mayor = (beta + raiz) / alfa, (beta - raiz) / alfa
    return (0.0, math.inf) if menor >= 0 else (max(mayor, 0.0), math.inf)
