"""Veredictos descriptivos por SKU y por categoria (3.4-A2, ADR-03-008, ADR-023).

La familia campeona de cada categoria y la instancia de cada familia en cada
SKU se eligen en el bloque de seleccion y quedan congeladas. Todo lo demas lee
solo el bloque de prueba. Por SKU se aplica la primera regla que se cumpla:
FALLO_TECNICO, NO_EVALUABLE, EVIDENCIA_INSUFICIENTE, VALIDADO, EXPLORATORIO.
Las reglas son descriptivas. Aparte, cada SKU con evidencia suficiente sobre
datos reales lleva la prueba HLN-DM de su campeon contra Seasonal Naive, con
BH entre los SKU de su categoria, y el intervalo de r que invierte esa prueba.
Cada categoria lleva la confirmacion de su campeon frente a las demas familias.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos import SkuClass
from pred_engine.forecasting.evaluaciones.calculo_errores import (
    FAMILIA_LINEA_BASE,
    VERSION_METRICAS,
    BloquesReserva,
    EvaluacionBloques,
    EvaluacionRetrospectivaError,
    MetricasCandidato,
)
from pred_engine.forecasting.evaluaciones.confirmacion import (
    CategoriaAConfirmar,
    ConfirmacionCategoria,
    confirmar_campeones,
)
from pred_engine.forecasting.evaluaciones.diebold_mariano import (
    PruebaDM,
    corregir_multiplicidad,
    intervalo_razon,
    probar_contra_linea_base,
)
from pred_engine.forecasting.evaluaciones.veredictos.contratos import (
    FAMILIA_FUNDACIONAL,
    POLITICA_INICIAL,
    VEREDICTOS,
    ModeloEvaluado,
    PoliticaSeleccion,
    ResultadoEvaluacion,
    ResumenCategoria,
    SeleccionCategoria,
    SignificanciaCategoria,
    Veredicto,
    VeredictoSku,
)
from pred_engine.forecasting.evaluaciones.veredictos.seleccion import (
    representante,
    seleccionar_categoria,
)

_logger = get_logger(__name__)


def emitir_veredictos(
    evaluaciones: Sequence[EvaluacionBloques],
    bloques: BloquesReserva,
    *,
    datos_sinteticos: bool | None,
    politica: PoliticaSeleccion = POLITICA_INICIAL,
    seleccion_previa: Sequence[SeleccionCategoria] = (),
) -> ResultadoEvaluacion:
    """Campeon congelado en el bloque de seleccion y evidencia del bloque de prueba.

    `datos_sinteticos` viene de la referencia de ingesta de M1: con datos
    sinteticos, aumentados o desconocidos ningun SKU pasa de EXPLORATORIO y no
    hay inferencia. Solo False declara datos reales. `seleccion_previa` es la
    seleccion de robustez de cada categoria (`seleccionar_previa`); una
    categoria que no aparece queda sin ella.
    """
    skus = [e.sku for e in evaluaciones]
    if len(set(skus)) != len(skus):
        _logger.error("Evaluacion rechazada: SKU repetido en la entrada")
        raise EvaluacionRetrospectivaError("cada SKU debe evaluarse una sola vez")
    if bloques.fraccion_seleccion != politica.fraccion_seleccion:
        raise EvaluacionRetrospectivaError(
            "los bloques no usan la fraccion de seleccion de la politica"
        )

    por_clase: dict[SkuClass, list[EvaluacionBloques]] = defaultdict(list)
    for evaluacion in evaluaciones:
        por_clase[evaluacion.sku_class].append(evaluacion)
    previas = {s.sku_class: s for s in seleccion_previa}

    selecciones: list[SeleccionCategoria] = []
    significancias: list[SignificanciaCategoria] = []
    por_categoria: list[list[VeredictoSku]] = []
    a_confirmar: list[CategoriaAConfirmar] = []
    for sku_class in sorted(por_clase):
        propias = por_clase[sku_class]
        seleccion = seleccionar_categoria(
            sku_class, [e.seleccion for e in propias], politica
        )
        propios, significancia = _significancia(
            [_veredicto_sku(e, seleccion, politica, datos_sinteticos) for e in propias],
            {e.sku: e for e in propias},
            politica,
        )
        selecciones.append(seleccion)
        significancias.append(significancia)
        por_categoria.append(propios)
        campeona = seleccion.familia_campeona
        familias = (*seleccion.medianas_r, FAMILIA_LINEA_BASE)
        a_confirmar.append(
            CategoriaAConfirmar(
                sku_class,
                campeona,
                tuple(f for f in sorted(seleccion.medianas_r) if f != campeona),
                {
                    e.sku: {
                        f: c for f in familias if (c := _instancia(e, f)) is not None
                    }
                    for e in propias
                },
            )
        )
    confirmaciones = confirmar_campeones(
        a_confirmar,
        bloques.origenes("prueba"),
        politica.confirmacion,
        datos_sinteticos=datos_sinteticos,
    )
    categorias = tuple(
        _resumen(
            seleccion,
            propios,
            significancia,
            confirmacion,
            _contendiente(seleccion, por_clase[seleccion.sku_class], politica),
            previas.get(seleccion.sku_class),
        )
        for seleccion, propios, significancia, confirmacion in zip(
            selecciones, por_categoria, significancias, confirmaciones, strict=True
        )
    )
    return ResultadoEvaluacion(
        version_politica=politica.version,
        version_metricas=VERSION_METRICAS,
        datos_sinteticos=datos_sinteticos,
        categorias=categorias,
        skus=tuple(v for propios in por_categoria for v in propios),
        bloques=bloques,
    )


def _instancia(evaluacion: EvaluacionBloques, familia: str) -> MetricasCandidato | None:
    """Instancia de la familia en el bloque de prueba, elegida en el de seleccion.

    Sin ventanas validas de seleccion queda la primera instancia de la familia:
    el bloque de prueba nunca decide cual se usa.
    """
    if familia == FAMILIA_LINEA_BASE:
        return evaluacion.prueba.linea_base
    elegido = representante(evaluacion.seleccion, familia)
    prueba = [c for c in evaluacion.prueba.candidatos if c.familia == familia]
    if elegido is None:
        return next(iter(prueba), None)
    return next(c for c in prueba if c.candidato_id == elegido.candidato_id)


def _veredicto_sku(
    evaluacion: EvaluacionBloques,
    seleccion: SeleccionCategoria,
    politica: PoliticaSeleccion,
    datos_sinteticos: bool | None,
) -> VeredictoSku:
    prueba = evaluacion.prueba
    familia = seleccion.familia_campeona
    fallback_desde = None
    if familia != FAMILIA_LINEA_BASE and not any(
        c.familia == familia for c in prueba.candidatos
    ):
        # El SKU no ordena familias: se conserva el ranking de clase.
        for siguiente in sorted(
            seleccion.medianas_r, key=lambda f: (seleccion.medianas_r[f], f)
        ):
            if siguiente != familia and any(
                c.familia == siguiente for c in prueba.candidatos
            ):
                fallback_desde = familia
                familia = siguiente
                break
    if familia == FAMILIA_LINEA_BASE:
        instancias = [prueba.linea_base]
    else:
        instancias = [c for c in prueba.candidatos if c.familia == familia]
    campeon = _instancia(evaluacion, familia)

    fallidos = [
        c
        for c in prueba.candidatos
        if (c.n_ventanas_totales > 0 or prueba.linea_base.n_ventanas_totales > 0)
        and c.n_pronosticos_validos == 0
    ]
    # Solo cuenta la falla de una familia que compite segun la politica.
    elegibles = politica.familias_elegibles[prueba.sku_class]
    comparacion_incompleta = bool(
        prueba.n_candidatos_fallidos or any(c.familia in elegibles for c in fallidos)
    )
    todos_fallaron = len(fallidos) == len(prueba.candidatos) and bool(
        prueba.candidatos or prueba.n_candidatos_fallidos
    )
    agregadas = None if campeon is None else campeon.agregadas
    razon = None if agregadas is None else agregadas.razon_sn
    n = 0 if campeon is None else campeon.n_ventanas_validas

    veredicto: Veredicto
    if (
        todos_fallaron
        or not instancias
        or (
            any(c.n_ventanas_totales > 0 for c in instancias)
            and all(c.n_pronosticos_validos == 0 for c in instancias)
        )
    ):
        veredicto = "FALLO_TECNICO"
        motivo = "la instancia campeona o todos los candidatos no produjeron pronostico"
    elif prueba.n_obs_validas_reserva == 0:
        veredicto = "NO_EVALUABLE"
        motivo = "no hay observaciones validas en el bloque de prueba"
    elif n < politica.n_min or prueba.reserva_toda_cero:
        veredicto = "EVIDENCIA_INSUFICIENTE"
        motivo = (
            "demanda real toda en cero en el bloque de prueba"
            if prueba.reserva_toda_cero
            else f"{n} ventanas validas de prueba (< n_min={politica.n_min})"
        )
    elif (
        datos_sinteticos is False
        and razon is not None
        and razon < 1.0
        and not (comparacion_incompleta)
    ):
        veredicto = "VALIDADO"
        motivo = "datos reales, ventanas suficientes y r < 1 (descriptivo)"
    else:
        veredicto = "EXPLORATORIO"
        motivo = _motivo_exploratorio(datos_sinteticos, razon, comparacion_incompleta)

    if fallback_desde is not None:
        motivo += (
            f"; fallback de familia {fallback_desde} a {familia}: "
            "sin instancia de la campeona de clase"
        )

    prueba_dm = _prueba_dm(
        campeon, prueba.linea_base, familia, veredicto, datos_sinteticos, politica
    )
    return VeredictoSku(
        sku=prueba.sku,
        sku_class=prueba.sku_class,
        veredicto=veredicto,
        familia_campeona=familia,
        candidato_campeon=None if campeon is None else campeon.candidato_id,
        n_ventanas=n,
        cobertura=0.0 if campeon is None else campeon.cobertura,
        razon_sn=razon,
        pierde_frente_a_linea_base=None if razon is None else razon >= 1.0,
        comparacion_incompleta=comparacion_incompleta,
        metricas_campeon=agregadas,
        metricas_linea_base=prueba.linea_base.agregadas,
        iqr_diferencia_mae=_iqr_diferencia(campeon, prueba.linea_base),
        diebold_mariano=prueba_dm,
        justificacion=f"{veredicto}: {motivo}",
        modelos_evaluados=tuple(
            ModeloEvaluado(
                candidato_id=c.candidato_id,
                familia=c.familia,
                modelo=c.modelo,
                razon_sn=None if c.agregadas is None else c.agregadas.razon_sn,
            )
            for c in (prueba.linea_base, *prueba.candidatos)
        ),
        intervalo_r=None
        if prueba_dm.p_valor is None or campeon is None
        else intervalo_razon(campeon, prueba.linea_base, nivel=1 - politica.alfa_dm),
    )


def _significancia(
    veredictos: Sequence[VeredictoSku],
    evaluaciones: Mapping[str, EvaluacionBloques],
    politica: PoliticaSeleccion,
) -> tuple[list[VeredictoSku], SignificanciaCategoria]:
    """BH dentro de la categoria, conteos BY e intervalos al nivel FCR."""
    alfa = politica.alfa_dm
    pruebas = {v.sku: v.diebold_mariano for v in veredictos}
    bh = corregir_multiplicidad(pruebas, alfa)
    by = corregir_multiplicidad(pruebas, alfa, metodo="fdr_by")
    probados = sum(1 for p in bh.values() if p.p_valor is not None)
    seleccionados = sum(1 for p in bh.values() if p.significativa)
    nivel_fcr = 1 - seleccionados * alfa / probados if seleccionados else None
    corregidos = []
    for v in veredictos:
        prueba = bh[v.sku]
        fcr = None
        if nivel_fcr is not None and prueba.significativa:
            campeon = _instancia(evaluaciones[v.sku], v.familia_campeona)
            assert campeon is not None
            fcr = intervalo_razon(
                campeon, evaluaciones[v.sku].prueba.linea_base, nivel=nivel_fcr
            )
        corregidos.append(replace(v, diebold_mariano=prueba, intervalo_r_fcr=fcr))
    return corregidos, SignificanciaCategoria(
        n_probados=probados,
        n_mejores=_conteo(bh, -1),
        n_peores=_conteo(bh, 1),
        n_mejores_by=_conteo(by, -1),
        n_peores_by=_conteo(by, 1),
        nivel_fcr=nivel_fcr,
    )


def _conteo(pruebas: Mapping[str, PruebaDM], signo: int) -> int:
    return sum(
        1
        for p in pruebas.values()
        if p.significativa and p.estadistico is not None and p.estadistico * signo > 0
    )


def _contendiente(
    seleccion: SeleccionCategoria,
    evaluaciones: Sequence[EvaluacionBloques],
    politica: PoliticaSeleccion,
) -> str | None:
    """La familia que la misma regla elige sin la fundacional (Chronos-2)."""
    if FAMILIA_FUNDACIONAL not in seleccion.medianas_r:
        return None
    sku_class = seleccion.sku_class
    resto = tuple(
        f for f in politica.familias_elegibles[sku_class] if f != FAMILIA_FUNDACIONAL
    )
    if not resto:
        return None
    sin_fundacional = replace(
        politica, familias_elegibles={**politica.familias_elegibles, sku_class: resto}
    )
    return seleccionar_categoria(
        sku_class, [e.seleccion for e in evaluaciones], sin_fundacional
    ).familia_campeona


def _motivo_exploratorio(
    datos_sinteticos: bool | None, razon: float | None, comparacion_incompleta: bool
) -> str:
    causas = []
    if datos_sinteticos is None:
        causas.append("origen de datos desconocido")
    if datos_sinteticos:
        causas.append("datos sinteticos o aumentados")
    if razon is None:
        causas.append("r no calculable")
    elif razon >= 1.0:
        causas.append(f"r={razon:.3f} >= 1 (resultado adverso)")
    if comparacion_incompleta:
        causas.append("comparacion incompleta: algun candidato fallo")
    return "; ".join(causas)


def _prueba_dm(
    campeon: MetricasCandidato | None,
    linea_base: MetricasCandidato,
    familia: str,
    veredicto: Veredicto,
    datos_sinteticos: bool | None,
    politica: PoliticaSeleccion,
) -> PruebaDM:
    # ADR-03-011: nunca sobre datos sinteticos, que inflan el poder de la prueba.
    if datos_sinteticos is None:
        return PruebaDM(0, 0, causa="origen_desconocido")
    if datos_sinteticos:
        return PruebaDM(0, 0, causa="datos_sinteticos")
    if familia == FAMILIA_LINEA_BASE:
        return PruebaDM(0, 0, causa="campeon_es_linea_base")
    if campeon is None or veredicto not in ("VALIDADO", "EXPLORATORIO"):
        return PruebaDM(0, 0, causa=f"veredicto:{veredicto}")
    return probar_contra_linea_base(
        campeon, linea_base, n_min_ventanas=politica.n_min_dm
    )


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
    seleccion: SeleccionCategoria,
    veredictos: Sequence[VeredictoSku],
    significancia: SignificanciaCategoria,
    confirmacion: ConfirmacionCategoria,
    contendiente: str | None,
    seleccion_previa: SeleccionCategoria | None,
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
        significancia=significancia,
        confirmacion=confirmacion,
        contendiente=contendiente,
        seleccion_previa=seleccion_previa,
    )
