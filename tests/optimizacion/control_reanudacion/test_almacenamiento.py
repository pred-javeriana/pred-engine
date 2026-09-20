"""Almacen 2.9-B1: round-trip, atomico, corrupto, version."""

from __future__ import annotations

from pathlib import Path

import pytest

from pred_engine.optimizacion.control_reanudacion import (
    SCHEMA_VERSION,
    ManifiestoAusenteError,
    ManifiestoCorrida,
    ManifiestoCorruptoError,
    VersionManifiestoError,
)
from pred_engine.optimizacion.control_reanudacion.almacenamiento import (
    AlmacenManifiestosFs,
)
from pred_engine.optimizacion.control_reanudacion.huella import calcular_huella

from .test_huella import _solicitud


def _manifiesto(run_id: str = "run-001", **kwargs: object) -> ManifiestoCorrida:
    solicitud = _solicitud(run_id=run_id)
    base: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "estado": "en_progreso",
        "familia": solicitud.familia,
        "sku_id": solicitud.sku_id,
        "seed": solicitud.seed,
        "metrica_objetivo": solicitud.metrica_objetivo,
        "configuracion_validacion": solicitud.validacion,
        "fingerprint_configuracion": calcular_huella(solicitud),
        "backend": "optuna",
        "backend_checkpoint": "backend.jsonl",
        "n_trials_objetivo": 10,
        "n_trials_finalizados": 1,
    }
    base.update(kwargs)
    return ManifiestoCorrida.model_validate(base)


def test_roundtrip(tmp_path: Path) -> None:
    almacen = AlmacenManifiestosFs(tmp_path)
    original = _manifiesto()
    almacen.guardar(original)
    assert almacen.existe("run-001")
    assert almacen.cargar("run-001") == original


def test_ausente(tmp_path: Path) -> None:
    with pytest.raises(ManifiestoAusenteError):
        AlmacenManifiestosFs(tmp_path).cargar("no-existe")


def test_corrupto(tmp_path: Path) -> None:
    destino = tmp_path / "run-001" / "manifiesto.json"
    destino.parent.mkdir(parents=True)
    destino.write_text("{no json", encoding="utf-8")
    with pytest.raises(ManifiestoCorruptoError):
        AlmacenManifiestosFs(tmp_path).cargar("run-001")


def test_version_no_soportada(tmp_path: Path) -> None:
    almacen = AlmacenManifiestosFs(tmp_path)
    almacen.guardar(_manifiesto())
    ruta = almacen.ruta_de("run-001")
    texto = ruta.read_text(encoding="utf-8").replace(
        '"schema_version": 1', '"schema_version": 99'
    )
    ruta.write_text(texto, encoding="utf-8")
    with pytest.raises(VersionManifiestoError, match="99"):
        almacen.cargar("run-001")


def test_replace_interrumpido_conserva_el_previo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    almacen = AlmacenManifiestosFs(tmp_path)
    previo = _manifiesto(estado="en_progreso", n_trials_finalizados=4)
    almacen.guardar(previo)

    def _boom(src: object, dst: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "pred_engine.optimizacion.control_reanudacion.almacenamiento.os.replace",
        _boom,
    )
    with pytest.raises(KeyboardInterrupt):
        almacen.guardar(_manifiesto(estado="completada", n_trials_finalizados=10))
    assert almacen.cargar("run-001") == previo


def test_tmp_huerfano_no_impide_leer(tmp_path: Path) -> None:
    almacen = AlmacenManifiestosFs(tmp_path)
    almacen.guardar(_manifiesto())
    tmp = almacen.ruta_de("run-001").with_name("manifiesto.json.tmp")
    tmp.write_text("{truncado", encoding="utf-8")
    assert almacen.cargar("run-001").n_trials_finalizados == 1
