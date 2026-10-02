"""Esquema del manifiesto M2 -> M3 (ADR-03-004): completo, estricto y sin defaults."""

from __future__ import annotations

import dataclasses
from datetime import date

import pytest
from pydantic import TypeAdapter, ValidationError
from tests._manifiestos import (
    CONFIGURACIONES,
    candidato,
    con_configuracion,
    sin_campo,
)

from pred_engine.comun.modelos.manifiesto_candidatos import (
    MODELO_POR_FAMILIA,
    Candidato,
    CandidatoClasico,
    CandidatoDL,
    CandidatoFundacional,
    CandidatoML,
    ConfigChronos2,
    ContextoParticion,
)
from pred_engine.comun.modelos.modelos_fundacionales.configuracion import (
    ConfiguracionFundacional,
)

_ADAPTADOR: TypeAdapter[Candidato] = TypeAdapter(Candidato)


@pytest.mark.parametrize(
    ("familia", "tipo"),
    [
        ("classical", CandidatoClasico),
        ("ml", CandidatoML),
        ("dl", CandidatoDL),
        ("foundation", CandidatoFundacional),
    ],
)
def test_entrada_valida_por_familia(familia: str, tipo: type) -> None:
    validado = _ADAPTADOR.validate_python(candidato(familia))
    assert isinstance(validado, tipo)
    assert validado.configuracion.model_dump() == CONFIGURACIONES[familia]


@pytest.mark.parametrize("familia", ["classical", "ml", "dl", "foundation"])
def test_cada_hiperparametro_es_obligatorio(familia: str) -> None:
    for campo in CONFIGURACIONES[familia]:
        incompleto = candidato(
            familia, configuracion=sin_campo(CONFIGURACIONES[familia], campo)
        )
        with pytest.raises(ValidationError) as capturado:
            _ADAPTADOR.validate_python(incompleto)
        assert capturado.value.errors()[0]["type"] == "missing"


@pytest.mark.parametrize("familia", ["classical", "ml", "dl", "foundation"])
def test_hiperparametro_sobrante_se_rechaza(familia: str) -> None:
    with pytest.raises(ValidationError) as capturado:
        _ADAPTADOR.validate_python(con_configuracion(familia, extra=1))
    assert capturado.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize(
    ("familia", "campo", "valor"),
    [
        ("classical", "p", 1.0),
        ("classical", "m", True),
        ("ml", "lags", 2.0),
        ("ml", "learning_rate", "0.1"),
        ("dl", "capas", True),
        ("dl", "dropout", "0"),
    ],
)
def test_tipo_incorrecto_se_rechaza(familia: str, campo: str, valor: object) -> None:
    with pytest.raises(ValidationError) as capturado:
        _ADAPTADOR.validate_python(con_configuracion(familia, **{campo: valor}))
    assert capturado.value.errors()[0]["type"].endswith("_type")


@pytest.mark.parametrize(
    ("familia", "campo", "valor"),
    [
        ("classical", "q", -1),
        ("ml", "subsample", 0.0),
        ("ml", "colsample_bytree", 1.5),
        ("ml", "learning_rate", float("inf")),
        ("dl", "capas", 1),
        ("dl", "dropout", 1.0),
        ("dl", "l2", 2.0),
    ],
)
def test_fuera_de_dominio_se_rechaza(familia: str, campo: str, valor: object) -> None:
    with pytest.raises(ValidationError):
        _ADAPTADOR.validate_python(con_configuracion(familia, **{campo: valor}))


def test_un_entero_es_aceptado_donde_se_espera_flotante() -> None:
    validado = _ADAPTADOR.validate_python(con_configuracion("ml", learning_rate=1))
    assert validado.configuracion.model_dump()["learning_rate"] == 1.0


@pytest.mark.parametrize(
    ("estacional", "valido"),
    [
        ({"m": 0}, True),
        ({"m": 12, "P": 1, "D": 1, "Q": 1, "tendencia": "n"}, True),
        ({"m": 1}, False),
        ({"m": 0, "P": 1}, False),
        ({"m": 0, "D": 1}, False),
    ],
)
def test_estacionalidad_sarima_coherente(estacional: dict, valido: bool) -> None:
    entrada = con_configuracion("classical", **estacional)
    if valido:
        _ADAPTADOR.validate_python(entrada)
    else:
        with pytest.raises(ValidationError) as capturado:
            _ADAPTADOR.validate_python(entrada)
        assert capturado.value.errors()[0]["loc"][-1] == "m"


@pytest.mark.parametrize(
    ("diferencias", "tendencia", "valido"),
    [
        ({}, "c", True),
        ({}, "n", True),
        ({"d": 1}, "n", True),
        ({"D": 1}, "n", True),
        ({"d": 1}, "c", False),
        ({"D": 1}, "c", False),
        ({}, "t", False),
    ],
)
def test_tendencia_sarima_coherente_con_la_diferenciacion(
    diferencias: dict, tendencia: str, valido: bool
) -> None:
    entrada = con_configuracion("classical", tendencia=tendencia, **diferencias)
    if valido:
        _ADAPTADOR.validate_python(entrada)
    else:
        with pytest.raises(ValidationError) as capturado:
            _ADAPTADOR.validate_python(entrada)
        assert capturado.value.errors()[0]["loc"][-1] == "tendencia"


@pytest.mark.parametrize(
    ("campo", "valor"),
    [("revision", "otra-revision"), ("cuantil_puntual", 0.9), ("max_contexto", 512)],
)
def test_chronos2_exige_la_configuracion_fijada(campo: str, valor: object) -> None:
    with pytest.raises(ValidationError) as capturado:
        _ADAPTADOR.validate_python(con_configuracion("foundation", **{campo: valor}))
    assert capturado.value.errors()[0]["loc"][-1] == campo


def test_esquema_chronos2_cubre_toda_la_configuracion_fundacional() -> None:
    campos = {campo.name for campo in dataclasses.fields(ConfiguracionFundacional)}
    assert set(ConfigChronos2.model_fields) == campos


def test_modelo_debe_corresponder_a_su_familia() -> None:
    with pytest.raises(ValidationError):
        _ADAPTADOR.validate_python(candidato("ml", modelo="sarima"))


def test_familia_desconocida_se_rechaza() -> None:
    with pytest.raises(ValidationError) as capturado:
        _ADAPTADOR.validate_python(candidato("ml") | {"familia": "croston"})
    assert capturado.value.errors()[0]["type"].startswith("union_tag")


@pytest.mark.parametrize("campo", ["semilla", "sku", "candidato_id", "sku_class"])
def test_identidad_y_semilla_obligatorias(campo: str) -> None:
    with pytest.raises(ValidationError):
        _ADAPTADOR.validate_python(sin_campo(candidato("dl"), campo))


@pytest.mark.parametrize(
    ("campo", "valor"), [("semilla", -1), ("sku", "  "), ("sku_class", "regular")]
)
def test_identidad_fuera_de_dominio(campo: str, valor: object) -> None:
    with pytest.raises(ValidationError):
        _ADAPTADOR.validate_python(candidato("dl", **{campo: valor}))


def test_catalogo_limitado_a_los_cuatro_modelos_del_hito() -> None:
    assert MODELO_POR_FAMILIA == {
        "classical": "sarima",
        "ml": "lightgbm",
        "dl": "mlp",
        "foundation": "chronos2",
    }


@pytest.mark.parametrize("fraccion", [0.0, 1.0])
def test_contexto_exige_fraccion_en_el_intervalo_abierto(fraccion: float) -> None:
    with pytest.raises(ValidationError):
        ContextoParticion(
            ingesta_ref_m1="x",
            t_corte_reserva=date(2024, 1, 1),
            fraccion_reserva=fraccion,
        )
