"""Huella 2.9-A2: determinismo, sensibilidad y motivos explicitos."""

from __future__ import annotations

import pytest

from pred_engine.optimizacion.control_reanudacion import (
    SCHEMA_VERSION,
    ConfiguracionOptimizador,
    ConfiguracionValidacion,
    IncompatibilidadCorridaError,
    ManifiestoCorrida,
    SolicitudCorrida,
)
from pred_engine.optimizacion.control_reanudacion.huella import (
    calcular_huella,
    motivos_incompatibilidad,
    verificar_compatibilidad,
)


def _validacion() -> ConfiguracionValidacion:
    return ConfiguracionValidacion(min_train=20, horizonte=1, paso=1, estacionalidad=7)


def _optimizador() -> ConfiguracionOptimizador:
    return ConfiguracionOptimizador(
        n_trials_objetivo=10,
        muestreador="tpe",
        min_ventanas=4,
        agregacion="media_recortada",
        proporcion_recorte=0.1,
        factor_reduccion=3,
        habilitar_poda_semantica=True,
    )


def _solicitud(**kwargs: object) -> SolicitudCorrida:
    base: dict[str, object] = {
        "run_id": "run-001",
        "familia": "classical",
        "sku_id": "105",
        "seed": 0,
        "metrica_objetivo": "mase",
        "validacion": _validacion(),
        "espacio_busqueda": {
            "parametros": [
                {"tipo": "entero", "nombre": "p", "bajo": 0, "alto": 5, "paso": 1}
            ],
            "condiciones": [],
            "restricciones": [],
        },
        "optimizador": _optimizador(),
        "n_observaciones": 60,
        "huella_serie": "abc",
        "backend": "optuna",
    }
    base.update(kwargs)
    return SolicitudCorrida.model_validate(base)


def _manifiesto_de(solicitud: SolicitudCorrida, **kwargs: object) -> ManifiestoCorrida:
    base: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": solicitud.run_id,
        "estado": "interrumpida",
        "familia": solicitud.familia,
        "sku_id": solicitud.sku_id,
        "seed": solicitud.seed,
        "metrica_objetivo": solicitud.metrica_objetivo,
        "configuracion_validacion": solicitud.validacion,
        "fingerprint_configuracion": calcular_huella(solicitud),
        "backend": solicitud.backend,
        "backend_checkpoint": "backend.jsonl",
        "n_trials_objetivo": solicitud.optimizador.n_trials_objetivo,
        "n_trials_finalizados": 2,
    }
    base.update(kwargs)
    return ManifiestoCorrida.model_validate(base)


def test_misma_configuracion_misma_huella() -> None:
    a = _solicitud(run_id="a")
    b = _solicitud(run_id="b")
    assert calcular_huella(a) == calcular_huella(b)


def test_run_id_no_participa_en_la_huella() -> None:
    assert calcular_huella(_solicitud(run_id="x")) == calcular_huella(
        _solicitud(run_id="y")
    )


def test_cambiar_seed_cambia_huella() -> None:
    assert calcular_huella(_solicitud(seed=0)) != calcular_huella(_solicitud(seed=1))


def test_cambiar_espacio_cambia_huella() -> None:
    otra = _solicitud(
        espacio_busqueda={
            "parametros": [
                {"tipo": "entero", "nombre": "p", "bajo": 0, "alto": 6, "paso": 1}
            ],
            "condiciones": [],
            "restricciones": [],
        }
    )
    assert calcular_huella(_solicitud()) != calcular_huella(otra)


def test_cambiar_serie_cambia_huella() -> None:
    assert calcular_huella(_solicitud(huella_serie="abc")) != calcular_huella(
        _solicitud(huella_serie="def")
    )


def test_compatibilidad_ok_no_lanza() -> None:
    solicitud = _solicitud()
    verificar_compatibilidad(solicitud, _manifiesto_de(solicitud))


def test_incompatibilidad_reporta_seed_y_huella() -> None:
    solicitud = _solicitud(seed=1)
    manifiesto = _manifiesto_de(_solicitud(seed=0))
    motivos = motivos_incompatibilidad(solicitud, manifiesto)
    assert "seed" in motivos
    assert "huella" in motivos
    with pytest.raises(IncompatibilidadCorridaError, match="seed") as captured:
        verificar_compatibilidad(solicitud, manifiesto)
    assert captured.value.motivos == motivos
