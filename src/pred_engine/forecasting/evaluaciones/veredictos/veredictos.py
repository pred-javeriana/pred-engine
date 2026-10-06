"""Veredictos descriptivos por SKU y por categoria (3.4-A2, ADR-03-008).

Por SKU se aplica la primera regla que se cumpla: FALLO_TECNICO,
NO_EVALUABLE, EVIDENCIA_INSUFICIENTE, VALIDADO, EXPLORATORIO. No se ejecutan
pruebas estadisticas: ningun veredicto afirma significancia.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos import SkuClass
from pred_engine.forecasting.evaluaciones.calculo_errores import (
    FAMILIA_LINEA_BASE,
    VERSION_METRICAS,
    EvaluacionRetrospectivaError,
    EvaluacionSku,
    MetricasCandidato,
)
from pred_engine.forecasting.evaluaciones.veredictos.contratos import (
    POLITICA_INICIAL,
    VEREDICTOS,
    ModeloEvaluado,
    PoliticaSeleccion,
    ResultadoEvaluacion,
    ResumenCategoria,
    SeleccionCategoria,
    Veredicto,
    VeredictoSku,
)
from pred_engine.forecasting.evaluaciones.veredictos.seleccion import (
    representante,
    seleccionar_categoria,
)

_logger = get_logger(__name__)


def emitir_veredictos(
    evaluaciones: Sequence[EvaluacionSku],
    *,
    datos_sinteticos: bool,
    politica: PoliticaSeleccion = POLITICA_INICIAL,
) -> ResultadoEvaluacion:
    """Seleccion por categoria y veredicto visible para cada SKU.

    `datos_sinteticos` viene de la referencia de ingesta de M1: con datos
    sinteticos o aumentados ningun SKU pasa de EXPLORATORIO.
    """
    skus = [e.sku for e in evaluaciones]
    if len(set(skus)) != len(skus):
        _logger.error("Evaluacion rechazada: SKU repetido en la entrada")
        raise EvaluacionRetrospectivaError("cada SKU debe evaluarse una sola vez")

    por_clase: dict[SkuClass, list[EvaluacionSku]] = defaultdict(list)
    for evaluacion in evaluaciones:
        por_clase[evaluacion.sku_class].append(evaluacion)

    categorias: list[ResumenCategoria] = []
    veredictos: list[VeredictoSku] = []
    for sku_class in sorted(por_clase):
        seleccion = seleccionar_categoria(sku_class, por_clase[sku_class], politica)
        propios = [
            _veredicto_sku(e, seleccion, politica, datos_sinteticos)
            for e in por_clase[sku_class]
        ]
        categorias.append(_resumen(seleccion, propios))
        veredictos.extend(propios)

    return ResultadoEvaluacion(
        version_politica=politica.version,
        version_metricas=VERSION_METRICAS,
        categorias=tuple(categorias),
        skus=tuple(veredictos),
    )


def _veredicto_sku(
    evaluacion: EvaluacionSku,
    seleccion: SeleccionCategoria,
    politica: PoliticaSeleccion,
    datos_sinteticos: bool,
) -> VeredictoSku:
    familia = seleccion.familia_campeona
    if familia == FAMILIA_LINEA_BASE:
        instancias = [evaluacion.linea_base]
        campeon: MetricasCandidato | None = evaluacion.linea_base
    else:
        instancias = [c for c in evaluacion.candidatos if c.familia == familia]
        campeon = representante(evaluacion, familia) or next(iter(instancias), None)

    fallidos = [c for c in evaluacion.candidatos if c.n_pronosticos_validos == 0]
    comparacion_incompleta = bool(evaluacion.n_candidatos_fallidos or fallidos)
    todos_fallaron = len(fallidos) == len(evaluacion.candidatos) and bool(
        evaluacion.candidatos or evaluacion.n_candidatos_fallidos
    )
    agregadas = None if campeon is None else campeon.agregadas
    razon = None if agregadas is None else agregadas.razon_sn
    n = 0 if campeon is None else campeon.n_ventanas_validas

    veredicto: Veredicto
    if todos_fallaron or all(c.n_pronosticos_validos == 0 for c in instancias):
        veredicto = "FALLO_TECNICO"
        motivo = "la instancia campeona o todos los candidatos no produjeron pronostico"
    elif evaluacion.n_obs_validas_reserva == 0:
        veredicto = "NO_EVALUABLE"
        motivo = "no hay observaciones validas en la reserva"
    elif n < politica.n_min or evaluacion.reserva_toda_cero:
        veredicto = "EVIDENCIA_INSUFICIENTE"
        motivo = (
            "demanda real toda en cero en la reserva"
            if evaluacion.reserva_toda_cero
            else f"{n} ventanas validas (< n_min={politica.n_min})"
        )
    elif (
        not datos_sinteticos
        and razon is not None
        and razon < 1.0
        and not (comparacion_incompleta)
    ):
        veredicto = "VALIDADO"
        motivo = "datos reales, ventanas suficientes y r < 1 (descriptivo)"
    else:
        veredicto = "EXPLORATORIO"
        motivo = _motivo_exploratorio(datos_sinteticos, razon, comparacion_incompleta)

    return VeredictoSku(
        sku=evaluacion.sku,
        sku_class=evaluacion.sku_class,
        veredicto=veredicto,
        familia_campeona=familia,
        candidato_campeon=None if campeon is None else campeon.candidato_id,
        n_ventanas=n,
        cobertura=0.0 if campeon is None else campeon.cobertura,
        razon_sn=razon,
        pierde_frente_a_linea_base=None if razon is None else razon >= 1.0,
        comparacion_incompleta=comparacion_incompleta,
        metricas_campeon=agregadas,
        metricas_linea_base=evaluacion.linea_base.agregadas,
        iqr_diferencia_mae=_iqr_diferencia(campeon, evaluacion.linea_base),
        justificacion=f"{veredicto}: {motivo}",
        modelos_evaluados=tuple(
            ModeloEvaluado(
                candidato_id=c.candidato_id,
                familia=c.familia,
                modelo=c.modelo,
                razon_sn=None if c.agregadas is None else c.agregadas.razon_sn,
            )
            for c in (evaluacion.linea_base, *evaluacion.candidatos)
        ),
    )


def _motivo_exploratorio(
    datos_sinteticos: bool, razon: float | None, comparacion_incompleta: bool
) -> str:
    causas = []
    if datos_sinteticos:
        causas.append("datos sinteticos o aumentados")
    if razon is None:
        causas.append("r no calculable")
    elif razon >= 1.0:
        causas.append(f"r={razon:.3f} >= 1 (resultado adverso)")
    if comparacion_incompleta:
        causas.append("comparacion incompleta: algun candidato fallo")
    return "; ".join(causas)


def _iqr_diferencia(
    campeon: MetricasCandidato | None, linea_base: MetricasCandidato
) -> float | None:
    if campeon is None:
        return None
    base = {v.origen: v.metricas.mae for v in linea_base.por_ventana}
    diferencias = [
        v.metricas.mae - base[v.origen] for v in campeon.por_ventana if v.origen in base
    ]
    if not diferencias:
        return None
    q75, q25 = np.percentile(diferencias, [75, 25])
    return float(q75 - q25)


def _resumen(
    seleccion: SeleccionCategoria, veredictos: Sequence[VeredictoSku]
) -> ResumenCategoria:
    conteos: dict[Veredicto, int] = {
        v: sum(1 for s in veredictos if s.veredicto == v) for v in VEREDICTOS
    }
    total = len(veredictos)
    porcentajes: dict[Veredicto, float] = {
        v: 100.0 * n / total for v, n in conteos.items()
    }
    razones = [s.razon_sn for s in veredictos if s.razon_sn is not None]
    return ResumenCategoria(
        sku_class=seleccion.sku_class,
        seleccion=seleccion,
        conteos=conteos,
        porcentajes=porcentajes,
        mediana_r=float(statistics.median(razones)) if razones else None,
        n_adversos=sum(1 for r in razones if r >= 1.0),
    )
