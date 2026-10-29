"""Prueba confirmatoria del campeon de clase en el bloque de prueba (ADR-023).

Para el campeon f* y cada competidora g de la categoria, con los SKU que tienen
ambas instancias y al menos ``n_min_ventanas`` ventanas comunes de prueba:

- x_i = log(RMSE_f*,i / RMSE_g,i) = 0.5 * log(sum_t L_f*,i(t) / sum_t L_g,i(t)).
  Una perdida cero es evidencia valida y no se descarta: x = -inf si solo el
  campeon no tiene error (razon 0), +inf si solo la competidora no lo tiene
  (razon infinita) y 0 si ninguno lo tiene (empate, razon 1);
- Wilcoxon de rangos con signo bilateral (exacto con n <= 50, ``scipy``); cuenta
  solo si la mediana de x es < 0, si no su p es 1;
- p de la categoria = maximo sobre las competidoras (interseccion-union) y Holm
  entre categorias;
- confirmado si el p de Holm es <= alfa y, para cada g, el intervalo bootstrap
  de la mediana de razones queda bajo 1. El bootstrap remuestrea bloques
  circulares de dias del bloque de prueba, los mismos dias para todos los SKU.

Con menos de ``n_min_skus`` SKU para alguna competidora la categoria queda
``sin_comparables``. Si el campeon es la linea base o los datos no son reales
declarados, ``no_aplica``: no hay inferencia sobre datos sinteticos (ADR-03-011).
Un valor que las perdidas cero dejan sin definir, como el promedio de una razon 0
y una infinita, queda en ``None`` y la causa ``perdida_cero:<campos>`` lo nombra.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from pred_engine.comun.logger import get_logger
from pred_engine.forecasting.evaluaciones.calculo_errores import (
    FAMILIA_LINEA_BASE,
    MetricasCandidato,
)
from pred_engine.forecasting.evaluaciones.confirmacion.contratos import (
    CategoriaAConfirmar,
    ComparacionFamilia,
    ConfirmacionCategoria,
    EfectoLineaBase,
    Intervalo,
    ParametrosConfirmacion,
)
from pred_engine.forecasting.evaluaciones.diebold_mariano import (
    corregir_multiplicidad,
    probar_pares,
)

_logger = get_logger(__name__)


def confirmar_campeones(
    categorias: Sequence[CategoriaAConfirmar],
    origenes: Iterable[pd.Timestamp],
    parametros: ParametrosConfirmacion,
    *,
    datos_sinteticos: bool | None,
) -> tuple[ConfirmacionCategoria, ...]:
    """Estado de cada campeon congelado, con Holm entre las categorias probadas.

    ``origenes`` son los origenes planeados del bloque de prueba, en orden: el
    bootstrap remuestrea sus posiciones.
    """
    posiciones = {pd.Timestamp(o): i for i, o in enumerate(origenes)}
    previas = [
        _confirmacion(c, posiciones, parametros, datos_sinteticos) for c in categorias
    ]
    probadas = [i for i, c in enumerate(previas) if c.p_iut is not None]
    if probadas:
        _, ajustados, _, _ = multipletests(
            [previas[i].p_iut for i in probadas], alpha=parametros.alfa, method="holm"
        )
        for i, p_holm in zip(probadas, ajustados, strict=True):
            previas[i] = _decidir(previas[i], float(p_holm), parametros.alfa)
    for c in previas:
        _logger.info(
            "Confirmacion categoria=%s campeona=%s estado=%s p_holm=%s causa=%s",
            c.sku_class,
            c.familia_campeona,
            c.estado,
            c.p_holm,
            c.causa,
        )
    return tuple(previas)


def _confirmacion(
    categoria: CategoriaAConfirmar,
    posiciones: Mapping[pd.Timestamp, int],
    parametros: ParametrosConfirmacion,
    datos_sinteticos: bool | None,
) -> ConfirmacionCategoria:
    campeona = categoria.familia_campeona
    causa = (
        "campeon_es_linea_base"
        if campeona == FAMILIA_LINEA_BASE
        else "origen_desconocido"
        if datos_sinteticos is None
        else "datos_sinteticos"
        if datos_sinteticos
        else None
    )
    if causa is not None:
        return ConfirmacionCategoria(
            categoria.sku_class,
            campeona,
            "no_aplica",
            causa,
            None,
            None,
            parametros.alfa,
            (),
            None,
        )
    efecto = _efecto(categoria, posiciones, parametros)
    comparaciones = tuple(
        _comparar(categoria, competidora, posiciones, parametros)
        for competidora in categoria.competidoras
    )
    p_competidoras = [c.p_iut for c in comparaciones if c.p_iut is not None]
    pocas = [c.competidora for c in comparaciones if c.p_iut is None]
    causa = (
        "sin_familias_competidoras"
        if not comparaciones
        else f"pocos_skus:{','.join(pocas)}"
        if pocas
        else None
    )
    # Sin causa, el estado final lo decide Holm entre categorias (_decidir).
    return ConfirmacionCategoria(
        categoria.sku_class,
        campeona,
        "sin_comparables" if causa else "no_confirmado",
        causa,
        None if causa else max(p_competidoras),
        None,
        parametros.alfa,
        comparaciones,
        efecto,
    )


def _decidir(
    confirmacion: ConfirmacionCategoria, p_holm: float, alfa: float
) -> ConfirmacionCategoria:
    sobre_uno = [
        c.competidora
        for c in confirmacion.comparaciones
        if c.ic_mediana is None or not c.ic_mediana.superior < 1
    ]
    causas = ([] if p_holm <= alfa else [f"p_holm>{alfa}"]) + (
        [f"intervalo_no_bajo_1:{','.join(sobre_uno)}"] if sobre_uno else []
    )
    return replace(
        confirmacion,
        estado="no_confirmado" if causas else "confirmado",
        causa="; ".join(causas) or None,
        p_holm=p_holm,
    )


def _efecto(
    categoria: CategoriaAConfirmar,
    posiciones: Mapping[pd.Timestamp, int],
    parametros: ParametrosConfirmacion,
) -> EfectoLineaBase:
    a, b, comunes, _ = _perdidas(
        categoria,
        categoria.familia_campeona,
        FAMILIA_LINEA_BASE,
        posiciones,
        parametros,
    )
    if not len(a):
        return EfectoLineaBase(0, None, None)
    centro = _mediana(_log_razones(a.sum(axis=1), b.sum(axis=1)))
    intervalo = _bootstrap(a, b, comunes, parametros)
    return EfectoLineaBase(
        len(a),
        _razon(centro),
        intervalo,
        _causa_cero(
            ("mediana_r", math.isnan(centro)), ("ic_mediana", intervalo is None)
        ),
    )


def _comparar(
    categoria: CategoriaAConfirmar,
    competidora: str,
    posiciones: Mapping[pd.Timestamp, int],
    parametros: ParametrosConfirmacion,
) -> ComparacionFamilia:
    a, b, comunes, skus = _perdidas(
        categoria, categoria.familia_campeona, competidora, posiciones, parametros
    )
    x = _log_razones(a.sum(axis=1), b.sum(axis=1))
    n = len(x)
    victorias = int((x < 0).sum())
    centro = _mediana(x) if n else math.nan
    sin_mediana = ("mediana_razon", bool(n) and math.isnan(centro))
    if n < parametros.n_min_skus:
        pocos = f"pocos_skus:{n}<{parametros.n_min_skus}"
        cero = _causa_cero(sin_mediana)
        return ComparacionFamilia(
            competidora,
            n,
            _razon(centro),
            None,
            None,
            None,
            victorias,
            None,
            None,
            None,
            None,
            None,
            None,
            pocos if cero is None else f"{pocos}; {cero}",
        )
    # scipy da a un |x| infinito el mayor rango: la prueba solo lee signos y rangos.
    wilcoxon = stats.wilcoxon(x)
    p_wilcoxon = float(wilcoxon.pvalue)
    ic_mediana = _bootstrap(a, b, comunes, parametros)
    hodges_lehmann = _hodges_lehmann(x, parametros.alfa)
    pruebas = corregir_multiplicidad(
        {
            sku: probar_pares(
                categoria.instancias[sku][categoria.familia_campeona],
                categoria.instancias[sku][competidora],
                n_min_ventanas=parametros.n_min_ventanas,
            )
            for sku in skus
        },
        parametros.alfa,
    )
    significativas = [
        p.estadistico
        for p in pruebas.values()
        if p.significativa and p.estadistico is not None
    ]
    hl, hl_inferior, hl_superior = hodges_lehmann or (math.nan,) * 3
    return ComparacionFamilia(
        competidora,
        n,
        _razon(centro),
        ic_mediana,
        _razon(hl),
        None
        if hodges_lehmann is None
        else Intervalo(float(np.exp(hl_inferior)), float(np.exp(hl_superior))),
        victorias,
        _wilson(victorias, n, parametros.alfa),
        float(wilcoxon.statistic),
        p_wilcoxon,
        # Una mediana sin definir no favorece al campeon.
        p_wilcoxon if centro < 0 else 1.0,
        sum(1 for e in significativas if e < 0),
        sum(1 for e in significativas if e > 0),
        _causa_cero(
            sin_mediana,
            ("ic_mediana", ic_mediana is None),
            ("hodges_lehmann", hodges_lehmann is None),
        ),
    )


def _perdidas(
    categoria: CategoriaAConfirmar,
    familia_a: str,
    familia_b: str,
    posiciones: Mapping[pd.Timestamp, int],
    parametros: ParametrosConfirmacion,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Perdidas por ventana (MSE) en las posiciones del bloque y sus ventanas comunes.

    Entran los SKU con ambas instancias y ``n_min_ventanas`` ventanas comunes,
    aunque su perdida total sea cero. Fuera de las ventanas comunes la perdida
    vale 0 y la mascara ``comunes`` es falsa.
    """
    filas_a, filas_b, filas_comunes, skus = [], [], [], []
    for sku in sorted(categoria.instancias):
        instancias = categoria.instancias[sku]
        a, b = instancias.get(familia_a), instancias.get(familia_b)
        if a is None or b is None:
            continue
        fila_a, fila_b, comunes = _fila(a, b, posiciones)
        if comunes.sum() < parametros.n_min_ventanas:
            continue
        filas_a.append(fila_a)
        filas_b.append(fila_b)
        filas_comunes.append(comunes)
        skus.append(sku)
    vacio = np.empty((0, len(posiciones)))
    return (
        np.array(filas_a) if filas_a else vacio,
        np.array(filas_b) if filas_b else vacio,
        np.array(filas_comunes) if filas_comunes else vacio.astype(bool),
        skus,
    )


def _fila(
    a: MetricasCandidato, b: MetricasCandidato, posiciones: Mapping[pd.Timestamp, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = {v.origen: v.metricas.rmse for v in b.por_ventana}
    fila_a, fila_b = np.zeros(len(posiciones)), np.zeros(len(posiciones))
    comunes = np.zeros(len(posiciones), dtype=bool)
    for v in a.por_ventana:
        if v.origen in base:
            i = posiciones[v.origen]
            fila_a[i], fila_b[i] = v.metricas.rmse**2, base[v.origen] ** 2
            comunes[i] = True
    return fila_a, fila_b, comunes


def _log_razones(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x = 0.5 * log(A / B) por SKU, con las perdidas totales cero explicitas.

    A = 0 < B da -inf (razon 0), B = 0 < A da +inf (razon infinita) y A = B = 0
    da 0 (empate). Sin epsilon: los bordes son exactos.
    """
    x = np.zeros(len(a))
    positivas = (a > 0) & (b > 0)
    x[positivas] = 0.5 * np.log(a[positivas] / b[positivas])
    x[(a == 0) & (b > 0)] = -np.inf
    x[(a > 0) & (b == 0)] = np.inf
    return x


def _mediana(x: np.ndarray) -> float:
    """Mediana de x; nan si sus dos valores centrales son -inf y +inf."""
    with np.errstate(invalid="ignore"):
        return float(np.median(x))


def _razon(log_razon: float) -> float | None:
    """exp(x), con 0 e infinito en los bordes; None si x no esta definido."""
    return None if math.isnan(log_razon) else float(np.exp(log_razon))


def _causa_cero(*campos: tuple[str, bool]) -> str | None:
    """``perdida_cero:<campos>`` con los campos que las perdidas cero no definen."""
    sin_definir = [nombre for nombre, falta in campos if falta]
    return f"perdida_cero:{','.join(sin_definir)}" if sin_definir else None


def _bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    comunes: np.ndarray,
    parametros: ParametrosConfirmacion,
) -> Intervalo | None:
    """Intervalo de la mediana de razones con bloques circulares de dias.

    Cada replica toma los mismos dias para todos los SKU, asi conserva la
    dependencia entre dias de cada SKU y los choques comunes entre SKU. Cada
    comparacion empieza del mismo generador: su intervalo no depende del orden.
    Un SKU entra en una replica si alguna de sus ventanas comunes sale
    remuestreada, con las razones 0, infinitas y empatadas de ``_log_razones``.
    ``None`` si alguna replica tiene una mediana sin definir o si un extremo cae
    entre una mediana -inf y una +inf.
    """
    rng = np.random.default_rng(parametros.semilla)
    t, bloque = a.shape[1], parametros.bloque_dias
    n_bloques = math.ceil(t / bloque)
    medianas = []
    for _ in range(parametros.replicas):
        inicios = rng.integers(0, t, size=n_bloques)
        indices = ((inicios[:, None] + np.arange(bloque)[None, :]).ravel() % t)[:t]
        conteo = np.bincount(indices, minlength=t).astype(float)
        presentes = comunes @ conteo > 0
        if presentes.any():
            medianas.append(
                _mediana(_log_razones((a @ conteo)[presentes], (b @ conteo)[presentes]))
            )
    if not medianas or any(math.isnan(m) for m in medianas):
        return None
    cola = 100 * parametros.alfa / 2
    extremos = _percentiles(np.array(medianas), [cola, 100 - cola])
    if np.isnan(extremos).any():
        return None
    inferior, superior = np.exp(extremos)
    return Intervalo(float(inferior), float(superior))


def _percentiles(valores: np.ndarray, percentiles: Sequence[float]) -> np.ndarray:
    """Percentiles lineales de numpy, con el limite si un vecino es infinito.

    Entre dos vecinos finitos el resultado es el de ``np.percentile``. Entre un
    vecino infinito y otro valor es ese infinito; entre -inf y +inf, nan.
    """
    with np.errstate(invalid="ignore"):
        lineal = np.percentile(valores, percentiles)
    bajo = np.percentile(valores, percentiles, method="lower")
    alto = np.percentile(valores, percentiles, method="higher")
    limite = np.where(
        np.isinf(bajo) & np.isinf(alto), np.nan, np.where(np.isinf(bajo), bajo, alto)
    )
    return np.where(
        bajo == alto,
        bajo,
        np.where(np.isinf(bajo) | np.isinf(alto), limite, lineal),
    )


def _hodges_lehmann(x: np.ndarray, alfa: float) -> tuple[float, float, float] | None:
    """Mediana de los promedios de Walsh e intervalo sin supuesto de distribucion.

    El intervalo va del (k+1)-esimo al (M-k)-esimo promedio, con M = n(n+1)/2 y k
    de la aproximacion normal del estadistico de rangos con signo. ``None`` con
    una razon 0 y una infinita: el promedio de -inf y +inf no esta definido.
    """
    if np.isneginf(x).any() and np.isposinf(x).any():
        return None
    n = len(x)
    i, j = np.triu_indices(n)
    walsh = np.sort((x[i] + x[j]) / 2.0)
    z = stats.norm.ppf(1 - alfa / 2)
    k = int(np.floor(n * (n + 1) / 4 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    k = max(k, 0)
    return float(np.median(walsh)), float(walsh[k]), float(walsh[len(walsh) - k - 1])


def _wilson(victorias: int, n: int, alfa: float) -> Intervalo:
    z = float(stats.norm.ppf(1 - alfa / 2))
    p = victorias / n
    d = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    mitad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return Intervalo(centro - mitad, centro + mitad)
