"""Ventanas completas, causalidad por origen y recuperacion de pronosticos 3.3."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter
from tests._manifiestos import candidato

from pred_engine.comun.modelos.manifiesto_candidatos import Candidato
from pred_engine.comun.reserva import ReserveCut
from pred_engine.forecasting.base import SeasonalNaiveStub
from pred_engine.forecasting.generador_pronosticos import (
    ConfiguracionPronosticos,
    SerieNoElegibleError,
    UnidadPronostico,
    generar_pronosticos,
    ventanas_reserva,
)

_CONFIGURACION = ConfiguracionPronosticos(horizonte=2, codigo_version="revision-1")
_CANDIDATO = TypeAdapter(Candidato).validate_python(candidato("classical"))
_CORTE = ReserveCut(
    t_star=pd.Timestamp("2024-01-16"),
    first_reserved=pd.Timestamp("2024-01-17"),
    last_observed=pd.Timestamp("2024-01-20"),
    reserved_days=4,
)


def _serie() -> pd.Series:
    return pd.Series(
        np.arange(1.0, 21.0), index=pd.date_range("2024-01-01", periods=20)
    )


def _fabrica(configuracion, *, seed):
    return SeasonalNaiveStub(season_length=1, seed=seed)


def _generar(serie=None, **opciones):
    return generar_pronosticos(
        _CANDIDATO,
        _serie() if serie is None else serie,
        _CORTE,
        opciones.pop("configuracion", _CONFIGURACION),
        fabricas=opciones.pop("fabricas", {"classical": _fabrica}),
        **opciones,
    )


def test_todos_los_origenes_diarios_con_horizonte_completo_y_trazabilidad():
    resultado = _generar(_serie().sample(frac=1, random_state=3))
    assert [p.origen.strftime("%Y-%m-%d") for p in resultado.serie.pronosticos] == [
        "2024-01-16",
        "2024-01-17",
        "2024-01-18",
    ]
    assert [
        p.fechas.strftime("%Y-%m-%d").tolist() for p in resultado.serie.pronosticos
    ] == [
        ["2024-01-17", "2024-01-18"],
        ["2024-01-18", "2024-01-19"],
        ["2024-01-19", "2024-01-20"],
    ]
    assert [p.valores.tolist() for p in resultado.serie.pronosticos] == [
        [16.0, 16.0],
        [17.0, 17.0],
        [18.0, 18.0],
    ]
    assert [(u.candidato.semilla, u.codigo_version) for u in resultado.unidades] == [
        (7, "revision-1"),
        (7, "revision-1"),
        (7, "revision-1"),
    ]
    assert all(
        u.emitido_en.utcoffset().total_seconds() == 0 for u in resultado.unidades
    )
    assert resultado.fallos == ()


def test_horizonte_igual_a_reserva_da_una_ventana_sin_recorte():
    resultado = _generar(configuracion=replace(_CONFIGURACION, horizonte=4))
    assert [p.valores.tolist() for p in resultado.serie.pronosticos] == [[16.0] * 4]
    assert resultado.serie.pronosticos[0].fechas[-1] == pd.Timestamp("2024-01-20")
    with pytest.raises(SerieNoElegibleError, match="INSUFFICIENT_RESERVE"):
        _generar(configuracion=replace(_CONFIGURACION, horizonte=5))


def test_ajuste_unico_no_consume_la_reserva_y_los_origenes_no_reajustan():
    class AjusteUnico(SeasonalNaiveStub):
        def fit(self, y):
            if self._fitted:
                raise ValueError("el modelo se ajusto mas de una vez")
            # El valor observado se publica como pronostico: un fit con la
            # reserva o mas de un ajuste cambia el resultado literal esperado.
            return super().fit(y)

    datos = _serie()
    datos.loc["2024-01-17":] = [101.0, 102.0, 103.0, 104.0]
    resultado = _generar(
        datos,
        fabricas={"classical": lambda config, *, seed: AjusteUnico(1, seed)},
    )
    assert [p.valores.tolist() for p in resultado.serie.pronosticos] == [
        [16.0, 16.0],
        [101.0, 101.0],
        [102.0, 102.0],
    ]
    assert resultado.fallos == ()


def test_causalidad_dorada_por_origen_y_filas_fuera_del_periodo():
    datos = _serie()
    datos.loc["2024-01-18":] = [900.0, 901.0, 902.0]
    datos.loc[pd.Timestamp("2024-01-21")] = 999.0
    resultado = _generar(datos)
    assert [p.valores.tolist() for p in resultado.serie.pronosticos] == [
        [16.0, 16.0],
        [17.0, 17.0],
        [900.0, 900.0],
    ]
    # Cambiar lo observado en el dia 18 solo afecta al origen 18, nunca a
    # los origenes 16 y 17. La observacion agregada fuera de la reserva no entra.
    assert datos.loc["2024-01-21"] == 999.0


def test_entrada_no_se_modifica_por_un_adaptador_que_muta_sus_arreglos():
    class Mutador(SeasonalNaiveStub):
        def fit(self, y):
            y[:] = 8.0
            return super().fit(y)

        def pronosticar(self, observadas, horizon):
            observadas[:] = 9.0
            return super().pronosticar(observadas, horizon)

    datos = _serie()
    resultado = _generar(
        datos, fabricas={"classical": lambda config, *, seed: Mutador(1, seed)}
    )
    assert [p.valores.tolist() for p in resultado.serie.pronosticos] == [
        [8.0, 8.0],
        [9.0, 9.0],
        [9.0, 9.0],
    ]
    assert datos.tolist() == list(np.arange(1.0, 21.0))


@pytest.mark.parametrize("salida", [np.nan, np.inf, "corta", "matriz", "excepcion"])
def test_salida_invalida_se_aisla_sin_ceros_y_el_siguiente_origen_continua(salida):
    class FallaInicial(SeasonalNaiveStub):
        def pronosticar(self, observadas, horizon):
            if len(observadas):
                return super().pronosticar(observadas, horizon)
            if salida == "corta":
                return np.array([1.0])
            if salida == "matriz":
                return np.ones((1, horizon))
            if salida == "excepcion":
                raise ValueError("no se pudo pronosticar")
            return np.full(horizon, salida)

    resultado = _generar(
        fabricas={"classical": lambda config, *, seed: FallaInicial(1, seed)}
    )
    assert np.isnan(resultado.serie.pronosticos[0].valores).all()
    assert [p.valores.tolist() for p in resultado.serie.pronosticos[1:]] == [
        [17.0, 17.0],
        [18.0, 18.0],
    ]
    assert [(f.sku, f.candidato_id, f.origen) for f in resultado.fallos] == [
        ("S1", "S1/classical/sarima", pd.Timestamp("2024-01-16"))
    ]
    assert resultado.fallos[0].codigo_error == (
        "pronostico_fallido" if salida == "excepcion" else "pronostico_invalido"
    )


def test_error_de_ajuste_registra_cada_unidad_y_otro_candidato_sigue():
    def falla(configuracion, *, seed):
        raise RuntimeError("ajuste no disponible")

    fallido = _generar(fabricas={"classical": falla})
    assert [f.codigo_error for f in fallido.fallos] == ["ajuste_fallido"] * 3
    assert all(np.isnan(p.valores).all() for p in fallido.serie.pronosticos)
    sano = _generar()
    assert sano.serie.pronosticos[-1].valores.tolist() == [18.0, 18.0]


def test_error_numerico_alto_y_pronostico_cero_se_conservan_como_evidencia():
    class Extremo(SeasonalNaiveStub):
        def pronosticar(self, observadas, horizon):
            return np.array([0.0, 1e12])

    resultado = _generar(
        fabricas={"classical": lambda config, *, seed: Extremo(1, seed)}
    )
    assert [p.valores.tolist() for p in resultado.serie.pronosticos] == [
        [0.0, 1e12]
    ] * 3
    assert resultado.fallos == ()


def test_falta_observacion_hasta_el_origen_solo_afecta_origenes_posteriores():
    datos = _serie().drop(pd.Timestamp("2024-01-17"))
    resultado = _generar(datos)
    assert resultado.serie.pronosticos[0].valores.tolist() == [16.0, 16.0]
    assert [f.codigo_error for f in resultado.fallos] == ["observaciones_ausentes"] * 2
    assert all(np.isnan(p.valores).all() for p in resultado.serie.pronosticos[1:])


@pytest.mark.parametrize(
    ("alteracion", "codigo"),
    [
        ("corta", "INSUFFICIENT_HISTORY"),
        ("hueco", "CALENDAR_GAPS"),
        ("duplicada", "CALENDAR_GAPS"),
        ("no_diaria", "CALENDAR_GAPS"),
        ("sin_fecha", "CALENDAR_GAPS"),
        ("infinita", "INVALID_HISTORY"),
        ("negativa", "INVALID_HISTORY"),
        ("texto", "INVALID_HISTORY"),
    ],
)
def test_exclusion_motivada_antes_del_ajuste(alteracion, codigo):
    datos = _serie()
    if alteracion == "corta":
        datos = datos.iloc[12:]
    elif alteracion == "hueco":
        datos = datos.drop(pd.Timestamp("2024-01-15"))
    elif alteracion == "duplicada":
        datos = pd.concat([datos, datos.iloc[:1]])
    elif alteracion == "no_diaria":
        datos.index += pd.Timedelta(hours=1)
    elif alteracion == "sin_fecha":
        datos = datos.reset_index(drop=True)
    elif alteracion == "infinita":
        datos.iloc[0] = np.inf
    elif alteracion == "negativa":
        datos.iloc[0] = -1.0
    elif alteracion == "texto":
        datos = datos.astype(object)
        datos.iloc[0] = "ilegible"
    with pytest.raises(SerieNoElegibleError, match=codigo) as error:
        _generar(datos)
    assert error.value.sku == "S1"
    assert error.value.codigo == codigo


@pytest.mark.parametrize("horizonte", [0, -1, True, 1.5])
def test_horizonte_invalido_no_se_ajusta(horizonte):
    with pytest.raises(ValueError, match="horizonte"):
        ConfiguracionPronosticos(horizonte=horizonte, codigo_version="v1")
    with pytest.raises(ValueError, match="horizonte"):
        ventanas_reserva(_CORTE, horizonte)


@pytest.mark.parametrize("cambios", [{"min_historia": 6}, {"codigo_version": " "}])
def test_protocolo_invalido_se_rechaza(cambios):
    with pytest.raises(ValueError):
        replace(_CONFIGURACION, **cambios)


@pytest.mark.parametrize(
    "cambios",
    [
        {"fraction": 0.3},
        {"reserved_days": 3},
        {"first_reserved": pd.Timestamp("2024-01-18")},
    ],
)
def test_corte_inconsistente_no_abre_el_generador(cambios):
    with pytest.raises(ValueError, match="corte de reserva inconsistente"):
        generar_pronosticos(
            _CANDIDATO, _serie(), replace(_CORTE, **cambios), _CONFIGURACION
        )


def test_caida_despues_de_confirmar_no_recalcula_ventanas_durables():
    confirmadas: list[UnidadPronostico] = []

    def confirmar_y_caer(unidad):
        confirmadas.append(unidad)
        if len(confirmadas) == 2:
            raise KeyboardInterrupt("caida despues de commit")

    with pytest.raises(KeyboardInterrupt):
        _generar(confirmar=confirmar_y_caer)
    datos = _serie()
    datos.loc["2024-01-17":] = [1000.0, 2000.0, 3000.0, 4000.0]
    nuevas: list[UnidadPronostico] = []
    resultado = _generar(datos, confirmadas=confirmadas, confirmar=nuevas.append)
    assert [p.valores.tolist() for p in resultado.serie.pronosticos] == [
        [16.0, 16.0],
        [17.0, 17.0],
        [2000.0, 2000.0],
    ]
    assert [u.pronostico.origen for u in nuevas] == [pd.Timestamp("2024-01-18")]
    assert resultado.unidades[0].emitido_en == confirmadas[0].emitido_en


def test_caida_antes_de_commit_reintenta_la_unidad_sin_marcador():
    def caer_antes_de_confirmar(unidad):
        raise OSError("disco no disponible")

    with pytest.raises(OSError, match="disco no disponible"):
        _generar(confirmar=caer_antes_de_confirmar)
    confirmadas: list[UnidadPronostico] = []
    resultado = _generar(confirmar=confirmadas.append)
    assert [u.pronostico.valores.tolist() for u in confirmadas] == [
        [16.0, 16.0],
        [17.0, 17.0],
        [18.0, 18.0],
    ]
    assert resultado.serie.pronosticos[-1].valores.tolist() == [18.0, 18.0]


def test_candidato_ya_confirmado_no_instancia_ni_ajusta_modelos():
    confirmadas = _generar().unidades

    def no_disponible(configuracion, *, seed):
        raise AssertionError("no debe reconstruir un candidato completado")

    resultado = _generar(confirmadas=confirmadas, fabricas={"classical": no_disponible})
    assert [p.valores.tolist() for p in resultado.serie.pronosticos] == [
        [16.0, 16.0],
        [17.0, 17.0],
        [18.0, 18.0],
    ]
    assert resultado.fallos == ()


@pytest.mark.parametrize(
    "cambio", ["version", "candidato", "fechas", "valores", "duplicado"]
)
def test_unidades_confirmadas_incompatibles_se_rechazan(cambio):
    original = _generar().unidades[0]
    if cambio == "version":
        unidad = replace(original, codigo_version="otra-revision")
    elif cambio == "candidato":
        unidad = replace(
            original, candidato=_CANDIDATO.model_copy(update={"semilla": 9})
        )
    elif cambio == "fechas":
        unidad = replace(
            original,
            pronostico=replace(
                original.pronostico, fechas=pd.date_range("2024-01-18", periods=2)
            ),
        )
    elif cambio == "valores":
        unidad = replace(
            original,
            pronostico=replace(original.pronostico, valores=np.array([np.nan, 1.0])),
        )
    else:
        unidad = original
    confirmadas = [unidad, unidad] if cambio == "duplicado" else [unidad]
    with pytest.raises(ValueError, match="unidad confirmada incompatible"):
        _generar(confirmadas=confirmadas)


def test_unidades_fallidas_confirmadas_no_se_reintentan():
    def falla(configuracion, *, seed):
        raise ValueError("ajuste fallido")

    previas = _generar(fabricas={"classical": falla}).unidades
    resultado = _generar(confirmadas=previas)
    assert [f.codigo_error for f in resultado.fallos] == ["ajuste_fallido"] * 3
    assert all(np.isnan(p.valores).all() for p in resultado.serie.pronosticos)
    # Una ejecucion nueva sin esos marcadores si conserva el candidato valido.
    assert _generar().serie.pronosticos[0].valores.tolist() == [16.0, 16.0]
