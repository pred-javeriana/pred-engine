"""Contratos 2.9-A1: manifiesto versionado, estados cerrados, sin Optuna."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from pred_engine.optimizacion.control_reanudacion import (
    SCHEMA_VERSION,
    ConfiguracionOptimizador,
    ConfiguracionValidacion,
    EstadoCorrida,
    ManifiestoCorrida,
    SolicitudCorrida,
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
        "espacio_busqueda": {"parametros": [{"tipo": "entero", "nombre": "p"}]},
        "optimizador": _optimizador(),
        "n_observaciones": 60,
        "huella_serie": "abc",
        "backend": "optuna",
    }
    base.update(kwargs)
    return SolicitudCorrida.model_validate(base)


def _manifiesto(**kwargs: object) -> ManifiestoCorrida:
    base: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "run-001",
        "estado": "nueva",
        "familia": "classical",
        "sku_id": "105",
        "seed": 0,
        "metrica_objetivo": "mase",
        "configuracion_validacion": _validacion(),
        "fingerprint_configuracion": "0" * 64,
        "backend": "optuna",
        "backend_checkpoint": None,
        "n_trials_objetivo": 10,
        "n_trials_finalizados": 0,
    }
    base.update(kwargs)
    return ManifiestoCorrida.model_validate(base)


def test_estados_cerrados_en_minusculas() -> None:
    assert tuple(e.value for e in EstadoCorrida) == (
        "nueva",
        "en_progreso",
        "interrumpida",
        "completada",
        "fallida",
    )


def test_manifiesto_rechaza_estado_desconocido() -> None:
    with pytest.raises(ValidationError):
        _manifiesto(estado="RUNNING")


def test_manifiesto_rechaza_schema_version_distinta_en_modelo_nuevo() -> None:
    # El modelo vigente solo acepta la version actual; versiones futuras
    # las rechaza el almacen, no Pydantic, via VersionManifiestoError.
    with pytest.raises(ValidationError):
        _manifiesto(schema_version=0)


def test_manifiesto_rechaza_campos_extra() -> None:
    with pytest.raises(ValidationError):
        ManifiestoCorrida(
            schema_version=SCHEMA_VERSION,
            run_id="r",
            estado="nueva",
            familia="classical",
            sku_id="105",
            seed=0,
            metrica_objetivo="mase",
            configuracion_validacion=_validacion(),
            fingerprint_configuracion="0" * 64,
            backend="optuna",
            n_trials_objetivo=1,
            n_trials_finalizados=0,
            optuna_study=1,  # type: ignore[call-arg]
        )


def test_manifiesto_json_roundtrip_sin_perdida() -> None:
    original = _manifiesto(
        estado="en_progreso",
        backend_checkpoint="backend.jsonl",
        n_trials_finalizados=3,
    )
    restaurado = ManifiestoCorrida.model_validate_json(original.model_dump_json())
    assert restaurado == original
    assert json.loads(original.model_dump_json())["estado"] == "en_progreso"


def test_solicitud_exige_run_id_no_vacio() -> None:
    with pytest.raises(ValidationError):
        _solicitud(run_id="   ")


def test_paquete_no_importa_optuna() -> None:
    import ast
    from pathlib import Path

    raiz = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "pred_engine"
        / "optimizacion"
        / "control_reanudacion"
    )
    prohibidos = ("optuna", "adaptador_optuna", "ejecutar_estudio", "walkforward")
    for path in raiz.glob("*.py"):
        arbol = ast.parse(path.read_text(encoding="utf-8"))
        nombres: set[str] = set()
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Import):
                nombres.update(alias.name for alias in nodo.names)
            elif isinstance(nodo, ast.ImportFrom) and nodo.module:
                nombres.add(nodo.module)
        unidos = " ".join(nombres)
        for palabra in prohibidos:
            assert palabra not in unidos, f"{path.name} importa {palabra}"
