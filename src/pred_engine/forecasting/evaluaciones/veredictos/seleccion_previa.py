"""Evidencia previa a t* para la seleccion de robustez (ADR-023, diseno B).

Convierte los pliegues walk-forward de L3, que solo usan historia hasta t*, en
la misma vista que la regla de seleccion lee de la reserva. La linea base es
Seasonal Naive ajustado con los datos de entrenamiento de cada pliegue. Las
familias optimizadas se puntuan en los pliegues que su HPO uso, asi que su
error L3 es optimista: esta seleccion se informa y nunca elige al campeon.
Como no lee la reserva, M3 la fija con el protocolo, antes de abrirla.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pred_engine.comun.dataclasses.validacion_temporal import ResultadoWalkForward
from pred_engine.comun.modelos import SkuClass
from pred_engine.forecasting.adaptador_candidatos import instanciar_linea_base
from pred_engine.forecasting.evaluaciones.calculo_errores import (
    FAMILIA_LINEA_BASE,
    MODELO_LINEA_BASE,
    EvaluacionSku,
    SerieCandidato,
    escala_q1,
    resumir_ventanas,
)
from pred_engine.forecasting.evaluaciones.veredictos.contratos import (
    POLITICA_INICIAL,
    PoliticaSeleccion,
    SeleccionCategoria,
)
from pred_engine.forecasting.evaluaciones.veredictos.seleccion import (
    seleccionar_categoria,
)


@dataclass(frozen=True, slots=True, eq=False)
class CandidatoPrevio:
    """Un candidato de M2 con su walk-forward de L3."""

    candidato_id: str
    familia: str
    modelo: str
    walk_forward: ResultadoWalkForward


def evaluacion_previa(
    sku: str,
    sku_class: SkuClass,
    t_star: pd.Timestamp,
    historia: np.ndarray,
    candidatos: Sequence[CandidatoPrevio],
) -> EvaluacionSku:
    """Metricas de los pliegues L3 del SKU; cada pliegue es una ventana.

    ``historia`` es la serie admisible hasta t* con que L3 armo los pliegues: el
    origen de un pliegue es la fecha de su ultima observacion de entrenamiento.
    """
    fechas = list(pd.date_range(end=t_star, periods=len(historia), freq="D"))
    base: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] = {}
    vistos: set[pd.Timestamp] = set()
    pares: list[dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]]] = []
    for candidato in candidatos:
        propios = {}
        for resultado in candidato.walk_forward.ventanas:
            origen = fechas[resultado.ventana.fin_train - 1]
            y_real = np.asarray(resultado.y_real, dtype=float)
            if origen not in vistos:
                vistos.add(origen)
                y_base = _linea_base(resultado.y_train, y_real)
                if y_base is not None:
                    base[origen] = (y_real, y_base)
            y_pred = np.asarray(resultado.y_pred, dtype=float)
            if (
                resultado.fallo is None
                and len(y_real)
                and y_pred.shape == y_real.shape
                and np.isfinite(y_pred).all()
                and np.isfinite(y_real).all()
            ):
                propios[origen] = (y_real, y_pred)
        pares.append(propios)
    escala = escala_q1(historia)
    linea_base = SerieCandidato(
        f"{sku}/{FAMILIA_LINEA_BASE}/{MODELO_LINEA_BASE}",
        FAMILIA_LINEA_BASE,
        MODELO_LINEA_BASE,
        (),
    )
    reales = np.concatenate([v[0] for v in base.values()]) if base else np.empty(0)
    return EvaluacionSku(
        sku=sku,
        sku_class=sku_class,
        linea_base=resumir_ventanas(
            linea_base,
            base,
            base,
            escala,
            n_ventanas_totales=len(base),
            n_pronosticos_validos=len(base),
        ),
        candidatos=tuple(
            resumir_ventanas(
                SerieCandidato(c.candidato_id, c.familia, c.modelo, ()),
                propios,
                base,
                escala,
                n_ventanas_totales=len(c.walk_forward.ventanas),
                n_pronosticos_validos=len(propios),
            )
            for c, propios in zip(candidatos, pares, strict=True)
        ),
        n_obs_validas_reserva=len(reales),
        reserva_toda_cero=bool(len(reales) and np.all(reales == 0)),
        n_candidatos_fallidos=0,
    )


def seleccionar_previa(
    previas: Sequence[EvaluacionSku],
    clases: Iterable[SkuClass],
    politica: PoliticaSeleccion = POLITICA_INICIAL,
) -> tuple[SeleccionCategoria, ...]:
    """La regla de seleccion sobre los pliegues L3, una por categoria y en orden.

    Una categoria sin SKU en `previas` queda sin SKU comparables.
    """
    return tuple(
        seleccionar_categoria(
            clase, [p for p in previas if p.sku_class == clase], politica
        )
        for clase in sorted(set(clases))
    )


def _linea_base(y_train: np.ndarray, y_real: np.ndarray) -> np.ndarray | None:
    """Seasonal Naive ajustado con el entrenamiento del pliegue; None si no aplica."""
    if not len(y_real) or not np.isfinite(y_real).all():
        return None
    try:
        modelo = instanciar_linea_base().fit(np.asarray(y_train, dtype=float))
    except ValueError:
        return None
    valores = np.asarray(modelo.pronosticar(np.empty(0), len(y_real)), dtype=float)
    return valores if np.isfinite(valores).all() else None
