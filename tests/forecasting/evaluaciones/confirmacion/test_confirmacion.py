"""Confirmacion del campeon congelado en el bloque de prueba (ADR-023)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests
from tests._evaluaciones import T_ESTRELLA, metricas_candidato

from pred_engine.forecasting.evaluaciones.calculo_errores import (
    FAMILIA_LINEA_BASE,
    MetricasCandidato,
)
from pred_engine.forecasting.evaluaciones.confirmacion import (
    CategoriaAConfirmar,
    confirmar_campeones,
)
from pred_engine.forecasting.evaluaciones.veredictos import POLITICA_INICIAL

PARAMETROS = POLITICA_INICIAL.confirmacion
ORIGENES = pd.date_range(T_ESTRELLA, periods=40, freq="D")
# Ejemplo trabajado de lumpy: 19 SKU favorecen al campeon y los dos de menor
# |x| (rangos 1 y 2) a la competidora, asi W+ = 3.
X_LUMPY = [0.0228, 0.0243, *(-0.03 - 0.01 * j for j in range(19))]


def _instancia(familia: str, rmse: float, n: int = 40) -> MetricasCandidato:
    return metricas_candidato(familia, 0.8, n=n, rmses=[rmse] * n)


def _categoria(
    sku_class: str,
    campeona: str,
    razones: Mapping[str, Sequence[float]],
    *,
    n: int = 40,
) -> CategoriaAConfirmar:
    """Por SKU i, RMSE_campeon / RMSE_g = exp(x_i) en cada ventana; SN vale 1.25."""
    n_skus = len(next(iter(razones.values())))
    return CategoriaAConfirmar(
        sku_class,  # type: ignore[arg-type]
        campeona,
        tuple(razones),
        {
            f"S{i}": {
                FAMILIA_LINEA_BASE: _instancia(FAMILIA_LINEA_BASE, 1.25),
                campeona: _instancia(campeona, 1.0, n),
                **{
                    g: _instancia(g, math.exp(-x[i]), n)
                    for g, x in razones.items()
                    if not math.isnan(x[i])
                },
            }
            for i in range(n_skus)
        },
    )


def _confirmar(*categorias, datos_sinteticos: bool | None = False, **cambios):
    return confirmar_campeones(
        categorias,
        ORIGENES,
        replace(PARAMETROS, **cambios),
        datos_sinteticos=datos_sinteticos,
    )


def test_ejemplo_trabajado_de_lumpy_contra_sarima() -> None:
    (confirmacion,) = _confirmar(
        _categoria("lumpy", "foundation", {"classical": X_LUMPY})
    )
    (fila,) = confirmacion.comparaciones
    assert (fila.competidora, fila.n_skus, fila.n_victorias) == ("classical", 21, 19)
    # Exacto: 2 * P(W+ <= 3) = 2 * 5 / 2^21 (subconjuntos {}, {1}, {2}, {3}, {1,2}).
    assert fila.estadistico_wilcoxon == 3.0
    assert fila.p_wilcoxon == pytest.approx(2 * 5 / 2**21)
    assert fila.p_iut == fila.p_wilcoxon
    x = np.array(X_LUMPY)
    assert fila.mediana_razon == pytest.approx(math.exp(np.median(x)))
    # Hodges-Lehmann: k = floor(115.5 - 1.95996 * 28.771) = 59 de M = 231.
    i, j = np.triu_indices(21)
    walsh = np.sort((x[i] + x[j]) / 2)
    assert fila.hodges_lehmann == pytest.approx(math.exp(np.median(walsh)))
    assert fila.ic_hodges_lehmann is not None
    assert fila.ic_hodges_lehmann.inferior == pytest.approx(math.exp(walsh[59]))
    assert fila.ic_hodges_lehmann.superior == pytest.approx(math.exp(walsh[171]))
    # Wilson de 19 de 21.
    assert fila.ic_victorias is not None
    assert fila.ic_victorias.inferior == pytest.approx(0.7109, abs=1e-4)
    assert fila.ic_victorias.superior == pytest.approx(0.9735, abs=1e-4)
    # Perdidas constantes por dia: el bootstrap no mueve la mediana.
    assert fila.ic_mediana is not None
    assert fila.ic_mediana.inferior == pytest.approx(fila.mediana_razon)
    assert fila.ic_mediana.superior == pytest.approx(fila.mediana_razon)
    assert (confirmacion.estado, confirmacion.causa) == ("confirmado", None)
    assert confirmacion.p_iut == confirmacion.p_holm == fila.p_wilcoxon
    efecto = confirmacion.efecto_linea_base
    assert efecto is not None and efecto.n_skus == 21
    assert efecto.mediana_r == pytest.approx(1 / 1.25)


def test_interseccion_union_y_holm_entre_categorias() -> None:
    fuerte = [-0.2 - 0.01 * j for j in range(12)]
    debil = [-0.05 + 0.002 * j for j in range(12)]
    confirmaciones = _confirmar(
        _categoria("smooth", "foundation", {"ml": fuerte, "dl": debil}),
        _categoria("erratic", "foundation", {"ml": fuerte}),
    )
    smooth, erratic = confirmaciones
    p = {c.competidora: c.p_wilcoxon for c in smooth.comparaciones}
    # La categoria vale lo de su competidora mas dificil.
    assert smooth.p_iut == max(p.values())
    assert erratic.p_iut == p["ml"]
    _, holm, _, _ = multipletests([smooth.p_iut, erratic.p_iut], method="holm")
    assert [smooth.p_holm, erratic.p_holm] == pytest.approx(list(holm))
    assert {c.estado for c in confirmaciones} == {"confirmado"}


def test_una_competidora_que_gana_la_mediana_no_confirma() -> None:
    gana_ml = [0.1 + 0.01 * j for j in range(12)]
    (confirmacion,) = _confirmar(
        _categoria("smooth", "foundation", {"ml": gana_ml}), replicas=200
    )
    (fila,) = confirmacion.comparaciones
    assert fila.p_wilcoxon is not None and fila.p_wilcoxon < 0.05
    assert fila.p_iut == 1.0
    assert confirmacion.estado == "no_confirmado"
    assert confirmacion.causa == "p_holm>0.05; intervalo_no_bajo_1:ml"
    # El campeon no cambia: solo se informa.
    assert confirmacion.familia_campeona == "foundation"


def test_pocos_skus_por_competidora_deja_la_categoria_sin_comparables() -> None:
    nueve = [-0.2] * 9
    confirmaciones = _confirmar(
        _categoria("smooth", "foundation", {"ml": nueve}),
        _categoria(
            "erratic", "foundation", {"ml": [-0.2 - 0.01 * j for j in range(12)]}
        ),
    )
    smooth, erratic = confirmaciones
    (fila,) = smooth.comparaciones
    assert (fila.n_skus, fila.causa, fila.p_iut) == (9, "pocos_skus:9<10", None)
    assert fila.mediana_razon == pytest.approx(math.exp(-0.2))
    assert (smooth.estado, smooth.causa) == ("sin_comparables", "pocos_skus:ml")
    assert smooth.p_holm is None
    # Holm solo cuenta las categorias probadas.
    assert erratic.p_holm == erratic.p_iut


def test_sku_con_pocas_ventanas_comunes_no_entra_en_la_comparacion() -> None:
    razones = [-0.2 - 0.01 * j for j in range(12)]
    categoria = _categoria("smooth", "foundation", {"ml": razones})
    corto = dict(categoria.instancias["S0"])
    corto["ml"] = _instancia("ml", 1.0, n=29)
    instancias = {**categoria.instancias, "S0": corto}
    (confirmacion,) = _confirmar(
        CategoriaAConfirmar("smooth", "foundation", ("ml",), instancias)
    )
    assert confirmacion.comparaciones[0].n_skus == 11


@pytest.mark.parametrize(
    ("campeona", "datos_sinteticos", "causa"),
    [
        (FAMILIA_LINEA_BASE, False, "campeon_es_linea_base"),
        ("foundation", True, "datos_sinteticos"),
        ("foundation", None, "origen_desconocido"),
    ],
)
def test_sin_inferencia_la_confirmacion_no_aplica(
    campeona: str, datos_sinteticos: bool | None, causa: str
) -> None:
    categoria = _categoria("smooth", "foundation", {"ml": [-0.2] * 12})
    (confirmacion,) = _confirmar(
        CategoriaAConfirmar("smooth", campeona, ("ml",), categoria.instancias),
        datos_sinteticos=datos_sinteticos,
    )
    assert (confirmacion.estado, confirmacion.causa) == ("no_aplica", causa)
    assert confirmacion.comparaciones == ()
    assert confirmacion.p_holm is None and confirmacion.efecto_linea_base is None


def test_sin_competidoras_no_hay_comparables() -> None:
    categoria = _categoria("lumpy", "classical", {"foundation": [-0.2] * 12})
    (confirmacion,) = _confirmar(
        CategoriaAConfirmar("lumpy", "classical", (), categoria.instancias)
    )
    assert (confirmacion.estado, confirmacion.causa) == (
        "sin_comparables",
        "sin_familias_competidoras",
    )
    assert confirmacion.efecto_linea_base is not None
    assert confirmacion.efecto_linea_base.n_skus == 12


def test_el_bootstrap_es_reproducible_y_no_depende_del_orden() -> None:
    generador = np.random.default_rng(3)
    ruido = generador.normal(0, 0.2, size=(12, 40))

    def instancias(orden: Sequence[str]) -> CategoriaAConfirmar:
        por_sku = {}
        for i in range(12):
            por_sku[f"S{i}"] = {
                FAMILIA_LINEA_BASE: _instancia(FAMILIA_LINEA_BASE, 1.25),
                "foundation": metricas_candidato(
                    "foundation", 0.8, rmses=list(np.exp(-0.2 + ruido[i]))
                ),
                "ml": _instancia("ml", 1.0),
                "dl": _instancia("dl", 1.1),
            }
        return CategoriaAConfirmar("smooth", "foundation", tuple(orden), por_sku)

    (primera,) = _confirmar(instancias(("ml", "dl")), replicas=300)
    (segunda,) = _confirmar(instancias(("dl", "ml")), replicas=300)
    intervalos = {c.competidora: c.ic_mediana for c in primera.comparaciones}
    assert intervalos == {c.competidora: c.ic_mediana for c in segunda.comparaciones}
    ic = intervalos["ml"]
    assert ic is not None and ic.inferior < ic.superior
