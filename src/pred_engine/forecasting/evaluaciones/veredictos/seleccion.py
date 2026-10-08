"""Regla de seleccion por categoria (3.4-A2, ADR-03-003 y ADR-03-007).

Por categoria (`sku_class`): mediana de r = RMSE_cand / RMSE_SN sobre el mismo
conjunto de SKUs para todas las familias elegibles; gana la menor mediana salvo
que no mejore a la linea base (compuerta) o empate con una familia mas simple.
Las familias elegibles las fija la politica; que M2 entregue o no una familia
se conoce antes de abrir la reserva, asi que excluirla no es una decision post hoc.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos import SkuClass
from pred_engine.forecasting.evaluaciones.calculo_errores import (
    FAMILIA_LINEA_BASE,
    EvaluacionSku,
    MetricasCandidato,
)
from pred_engine.forecasting.evaluaciones.veredictos.contratos import (
    MotivoSeleccion,
    PoliticaSeleccion,
    SeleccionCategoria,
)

_logger = get_logger(__name__)


def representante(evaluacion: EvaluacionSku, familia: str) -> MetricasCandidato | None:
    """Configuracion de la familia con menor RMSE en la reserva."""
    opciones = [
        c
        for c in evaluacion.candidatos
        if c.familia == familia and c.agregadas is not None
    ]
    return min(opciones, key=_rmse, default=None)


def seleccionar_categoria(
    sku_class: SkuClass,
    evaluaciones: Sequence[EvaluacionSku],
    politica: PoliticaSeleccion,
) -> SeleccionCategoria:
    # Compiten las familias elegibles que M2 entrego para esta categoria. Un SKU
    # sin r para alguna de ellas sale del conjunto comun.
    elegibles = politica.familias_elegibles[sku_class]
    entregadas = {c.familia for e in evaluaciones for c in e.candidatos}
    familias = sorted(f for f in elegibles if f in entregadas)
    familias_excluidas = {f: "no_entregada" for f in elegibles if f not in entregadas}
    familias_excluidas |= {f: "no_elegible" for f in entregadas if f not in elegibles}
    razones: dict[str, dict[str, float]] = {}
    excluidos: dict[str, str] = {}
    for evaluacion in evaluaciones:
        por_familia: dict[str, float] = {}
        for familia in familias:
            elegido = representante(evaluacion, familia)
            razon = None if elegido is None else _razon(elegido)
            if razon is None:
                excluidos[evaluacion.sku] = f"sin_razon_calculable:{familia}"
                break
            por_familia[familia] = razon
        else:
            razones[evaluacion.sku] = por_familia

    comparables = tuple(sorted(razones))
    if not familias or not comparables:
        return _seleccion(
            sku_class,
            FAMILIA_LINEA_BASE,
            {},
            False,
            "sin_skus_comparables",
            comparables,
            excluidos,
            familias_excluidas,
        )

    medianas = {
        f: float(statistics.median(razones[s][f] for s in comparables))
        for f in familias
    }
    mejor = min(medianas.values())
    if mejor >= 1.0:
        # Ninguna categoria queda con un campeon peor que la linea base.
        return _seleccion(
            sku_class,
            FAMILIA_LINEA_BASE,
            medianas,
            True,
            "compuerta_linea_base",
            comparables,
            excluidos,
            familias_excluidas,
        )
    # El desempate solo considera familias que tambien mejoran la linea base.
    empatadas = [
        f
        for f in familias
        if medianas[f] - mejor < politica.tolerancia_empate and medianas[f] < 1.0
    ]
    ganadora = min(empatadas, key=lambda f: _simplicidad(f, politica))
    motivo: MotivoSeleccion = (
        "empate_por_simplicidad" if len(empatadas) > 1 else "menor_mediana"
    )
    return _seleccion(
        sku_class,
        ganadora,
        medianas,
        False,
        motivo,
        comparables,
        excluidos,
        familias_excluidas,
    )


def _seleccion(
    sku_class: SkuClass,
    familia: str,
    medianas: dict[str, float],
    adverso: bool,
    motivo: MotivoSeleccion,
    comparables: tuple[str, ...],
    excluidos: dict[str, str],
    familias_excluidas: dict[str, str],
) -> SeleccionCategoria:
    _logger.info(
        "Seleccion categoria=%s campeona=%s motivo=%s comparables=%d excluidos=%d"
        " familias_excluidas=%s",
        sku_class,
        familia,
        motivo,
        len(comparables),
        len(excluidos),
        familias_excluidas,
    )
    return SeleccionCategoria(
        sku_class=sku_class,
        familia_campeona=familia,
        medianas_r=medianas,
        adverso=adverso,
        motivo=motivo,
        skus_comparables=comparables,
        excluidos=excluidos,
        familias_excluidas=familias_excluidas,
    )


def _simplicidad(familia: str, politica: PoliticaSeleccion) -> tuple[int, str]:
    orden = politica.orden_simplicidad
    return (orden.index(familia) if familia in orden else len(orden), familia)


def _rmse(candidato: MetricasCandidato) -> float:
    assert candidato.agregadas is not None
    return candidato.agregadas.rmse


def _razon(candidato: MetricasCandidato) -> float | None:
    return None if candidato.agregadas is None else candidato.agregadas.razon_sn
