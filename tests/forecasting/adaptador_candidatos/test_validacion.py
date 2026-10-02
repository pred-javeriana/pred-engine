"""Validacion 3.2-A1/A2: rechazo fail-closed del lote y aislamiento por candidato."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import pytest
from tests._dobles import PipelineFalso
from tests._manifiestos import (
    CONTEXTO,
    EMITIDO_EN,
    candidato,
    con_configuracion,
    manifiesto,
    sin_campo,
)

from pred_engine.comun.modelos.modelos_fundacionales import (
    ModeloFundacionalNoDisponibleError,
)
from pred_engine.forecasting.adaptador_candidatos import (
    FABRICAS,
    LoteRechazadoError,
    validar_manifiesto,
)

_TODOS = ("classical", "ml", "dl", "foundation")


class _Cargador:
    def __init__(self, error: Exception | None = None) -> None:
        self.llamadas = 0
        self._error = error

    def __call__(self, configuracion: object) -> PipelineFalso:
        self.llamadas += 1
        if self._error is not None:
            raise self._error
        return PipelineFalso()


class _CapturaLogger:
    def __init__(self) -> None:
        self.errores: list[str] = []

    def info(self, mensaje: str, *args: object) -> None:
        pass

    def error(self, mensaje: str, *args: object) -> None:
        self.errores.append(mensaje % args if args else mensaje)


def _validar(contenido: str, **kwargs: Any):
    kwargs.setdefault("esperado", CONTEXTO)
    kwargs.setdefault("skus_panel", {"S1", "S2"})
    kwargs.setdefault("cargar_fundacional", _Cargador())
    return validar_manifiesto(contenido, **kwargs)


def _rechazo(contenido: str | bytes, **kwargs: Any) -> LoteRechazadoError:
    with pytest.raises(LoteRechazadoError) as capturado:
        _validar(contenido, **kwargs)  # type: ignore[arg-type]
    return capturado.value


def test_manifiesto_valido_conserva_todos_los_candidatos() -> None:
    resultado = _validar(manifiesto(*(candidato(f) for f in _TODOS)))
    assert [c.familia for c in resultado.candidatos] == list(_TODOS)
    assert resultado.fallos == ()
    assert resultado.contexto == CONTEXTO
    assert resultado.run_id_m2 == "m2-corrida-1"


def test_acepta_bytes() -> None:
    contenido = manifiesto(candidato("ml")).encode("utf-8")
    assert len(_validar(contenido).candidatos) == 1  # type: ignore[arg-type]


# --- Nivel lote (fail-closed) ---------------------------------------------


@pytest.mark.parametrize(
    "contenido",
    [
        "{no es json",
        b"\xff\xfe",
        "[]",
        manifiesto(candidato("ml"), run_id_m2=""),
        manifiesto(candidato("ml"), campo_extra=1),
        manifiesto(candidato("ml"), emitido_en="2024-03-01T12:00:00"),
        manifiesto(candidato("ml"), candidatos=[["no", "es", "objeto"]]),
        manifiesto(sin_campo(candidato("ml"), "candidato_id")),
        manifiesto(candidato("ml", sku="   ")),
    ],
)
def test_manifiesto_ilegible_rechaza_el_lote(contenido: str) -> None:
    assert _rechazo(contenido).motivo == "manifiesto_ilegible"


@pytest.mark.parametrize("version", [2, 0, None, "1"])
def test_version_no_soportada_rechaza_el_lote(version: object) -> None:
    contenido = manifiesto(candidato("ml"), schema_version=version)
    assert _rechazo(contenido).motivo == "version_no_soportada"


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        ("ingesta_ref_m1", "b" * 64),
        ("t_corte_reserva", "2024-02-01"),
        ("fraccion_reserva", 0.3),
    ],
)
def test_contexto_distinto_de_la_particion_rechaza_el_lote(
    campo: str, valor: object
) -> None:
    contexto = json.loads(CONTEXTO.model_dump_json()) | {campo: valor}
    error = _rechazo(manifiesto(candidato("ml"), contexto=contexto))
    assert error.motivo == "contexto_incompatible"
    assert campo in error.detalle


def test_handoff_emitido_tras_abrir_la_reserva_rechaza_el_lote() -> None:
    contenido = manifiesto(candidato("ml"))
    for apertura in (EMITIDO_EN, EMITIDO_EN - timedelta(seconds=1)):
        error = _rechazo(contenido, reserva_abierta_en=apertura)
        assert error.motivo == "handoff_posterior_a_reserva"
    resultado = _validar(contenido, reserva_abierta_en=EMITIDO_EN + timedelta(hours=1))
    assert len(resultado.candidatos) == 1


def test_candidato_id_duplicado_rechaza_el_lote() -> None:
    contenido = manifiesto(
        candidato("ml"), candidato("dl", candidato_id="S1/ml/lightgbm")
    )
    error = _rechazo(contenido)
    assert error.motivo == "candidato_id_duplicado"
    assert "S1/ml/lightgbm" in error.detalle


def test_sku_fuera_del_panel_rechaza_el_lote() -> None:
    error = _rechazo(manifiesto(candidato("ml", sku="S9")))
    assert error.motivo == "sku_fuera_del_panel"
    assert "S9" in error.detalle


def test_error_no_clasificado_cierra_el_lote() -> None:
    cargador = _Cargador(error=RuntimeError("fallo inesperado"))
    error = _rechazo(manifiesto(candidato("foundation")), cargar_fundacional=cargador)
    assert error.motivo == "error_no_clasificado"
    assert isinstance(error.__cause__, RuntimeError)


def test_rechazo_de_lote_queda_registrado(monkeypatch: pytest.MonkeyPatch) -> None:
    captura = _CapturaLogger()
    monkeypatch.setattr(
        "pred_engine.forecasting.adaptador_candidatos.validacion._logger", captura
    )
    _rechazo(manifiesto(candidato("ml", sku="S9")))
    assert len(captura.errores) == 1
    assert "sku_fuera_del_panel" in captura.errores[0]


# --- Nivel candidato (aislado) --------------------------------------------


@pytest.mark.parametrize(
    ("malo", "campo", "codigo"),
    [
        (
            con_configuracion("ml", lags=2.0),
            "configuracion.lags",
            "tipo_incorrecto",
        ),
        (
            candidato(
                "ml",
                configuracion=sin_campo(con_configuracion("ml")["configuracion"], "m"),
            ),
            "configuracion.m",
            "campo_faltante",
        ),
        (
            con_configuracion("ml", sobrante=1),
            "configuracion.sobrante",
            "campo_sobrante",
        ),
        (
            con_configuracion("ml", subsample=0.0),
            "configuracion.subsample",
            "fuera_de_dominio",
        ),
        (sin_campo(candidato("ml"), "semilla"), "semilla", "semilla_ausente"),
        (candidato("ml") | {"familia": "croston"}, "familia", "modelo_no_registrado"),
        (candidato("ml", modelo="xgboost"), "modelo", "modelo_no_registrado"),
    ],
)
def test_candidato_invalido_no_descarta_a_los_demas(
    malo: dict, campo: str, codigo: str
) -> None:
    malo = malo | {"candidato_id": "S1/malo"}
    resultado = _validar(manifiesto(candidato("classical"), malo, candidato("dl")))
    assert [c.familia for c in resultado.candidatos] == ["classical", "dl"]
    (fallo,) = resultado.fallos
    assert (fallo.sku, fallo.candidato_id) == ("S1", "S1/malo")
    assert (fallo.campo, fallo.codigo_error) == (campo, codigo)


def test_mensaje_identifica_sku_modelo_y_campo() -> None:
    malo = con_configuracion("dl", capas=1) | {"sku": "S2", "candidato_id": "S2/dl"}
    (fallo,) = _validar(manifiesto(malo)).fallos
    for fragmento in ("sku=S2", "modelo=mlp", "campo=configuracion.capas"):
        assert fragmento in fallo.mensaje
    assert fallo.modelo == "mlp"


def test_modelo_ilegible_queda_como_desconocido() -> None:
    malo = candidato("ml", modelo=3)
    (fallo,) = _validar(manifiesto(malo)).fallos
    assert fallo.modelo is None
    assert fallo.codigo_error == "modelo_no_registrado"


def test_familia_deshabilitada_en_el_registro_falla_aislada() -> None:
    fabricas = {f: fabrica for f, fabrica in FABRICAS.items() if f != "dl"}
    resultado = _validar(
        manifiesto(candidato("ml"), candidato("dl")), fabricas=fabricas
    )
    assert [c.familia for c in resultado.candidatos] == ["ml"]
    (fallo,) = resultado.fallos
    assert (fallo.campo, fallo.codigo_error) == ("familia", "modelo_no_registrado")


def test_pesos_no_disponibles_fallan_solo_los_fundacionales() -> None:
    cargador = _Cargador(error=ModeloFundacionalNoDisponibleError("sin pesos"))
    contenido = manifiesto(
        candidato("foundation"),
        candidato("ml"),
        candidato("foundation", sku="S2"),
    )
    resultado = _validar(contenido, cargar_fundacional=cargador)
    assert [c.familia for c in resultado.candidatos] == ["ml"]
    assert [f.codigo_error for f in resultado.fallos] == ["pesos_no_disponibles"] * 2
    assert {f.sku for f in resultado.fallos} == {"S1", "S2"}
    assert cargador.llamadas == 1


def test_sin_candidatos_fundacionales_no_carga_pesos() -> None:
    cargador = _Cargador()
    _validar(manifiesto(candidato("ml")), cargar_fundacional=cargador)
    assert cargador.llamadas == 0


def test_cada_fallo_de_candidato_queda_registrado(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captura = _CapturaLogger()
    monkeypatch.setattr(
        "pred_engine.forecasting.adaptador_candidatos.validacion._logger", captura
    )
    contenido = manifiesto(
        con_configuracion("ml", lags=0) | {"candidato_id": "a"},
        con_configuracion("dl", dropout=1.0) | {"candidato_id": "b"},
        candidato("classical"),
    )
    resultado = _validar(contenido)
    assert len(resultado.fallos) == 2
    assert len(captura.errores) == 2
    assert all("candidato_id=" in e and "codigo=" in e for e in captura.errores)


def test_apertura_de_reserva_sin_zona_horaria_cierra_el_lote() -> None:
    # Una fecha naive no es comparable con `emitido_en` (aware): no se adivina.
    error = _rechazo(
        manifiesto(candidato("ml")), reserva_abierta_en=datetime(2024, 3, 2)
    )
    assert error.motivo == "error_no_clasificado"
