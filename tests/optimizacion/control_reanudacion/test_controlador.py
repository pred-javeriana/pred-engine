"""Controlador 2.9-C1: crear, reanudar, rechazar, no reabrir completada."""

from __future__ import annotations

from pathlib import Path

import pytest

from pred_engine.optimizacion.control_reanudacion import (
    AlmacenManifiestosFs,
    ControladorReanudacion,
    CorridaFallidaError,
    EstadoCorrida,
    IncompatibilidadCorridaError,
)

from .test_huella import _solicitud


class FakePuerto:
    def __init__(self) -> None:
        self.creaciones = 0
        self.persistencias = 0
        self.restauraciones = 0
        self._store: dict[str, list[int]] = {}
        self._n = 0

    def crear(self) -> object:
        self.creaciones += 1
        handle: list[int] = []
        return handle

    def persistir(self, handle: object) -> str:
        self.persistencias += 1
        self._n += 1
        clave = f"ckpt-{self._n}"
        self._store[clave] = list(handle)  # type: ignore[arg-type]
        return clave

    def restaurar(self, referencia: str) -> object:
        self.restauraciones += 1
        return list(self._store.get(referencia, []))


def _controlador(tmp_path: Path) -> tuple[ControladorReanudacion, FakePuerto]:
    puerto = FakePuerto()
    ctrl = ControladorReanudacion(AlmacenManifiestosFs(tmp_path), puerto)
    return ctrl, puerto


def test_crear_cuando_no_hay_estado(tmp_path: Path) -> None:
    ctrl, puerto = _controlador(tmp_path)
    sesion = ctrl.abrir(_solicitud())
    assert sesion.manifiesto.estado is EstadoCorrida.EN_PROGRESO
    assert puerto.creaciones == 1
    assert sesion.manifiesto.n_trials_finalizados == 0


def test_reanudar_interrumpida_compatible(tmp_path: Path) -> None:
    ctrl, puerto = _controlador(tmp_path)
    sesion = ctrl.abrir(_solicitud())
    sesion.handle.append(7)  # type: ignore[union-attr]
    ctrl.interrumpir(sesion, n_trials_finalizados=2)
    otra = ctrl.abrir(_solicitud())
    assert otra.manifiesto.estado is EstadoCorrida.EN_PROGRESO
    assert otra.handle == [7]
    assert puerto.restauraciones == 1


def test_en_progreso_es_reanudable(tmp_path: Path) -> None:
    ctrl, _puerto = _controlador(tmp_path)
    sesion = ctrl.abrir(_solicitud())
    ctrl.checkpoint(sesion, n_trials_finalizados=3)
    otra = ctrl.abrir(_solicitud())
    assert otra.manifiesto.estado is EstadoCorrida.EN_PROGRESO
    assert otra.manifiesto.n_trials_finalizados == 3


def test_incompatibilidad_no_muta_persistido(tmp_path: Path) -> None:
    ctrl, _ = _controlador(tmp_path)
    sesion = ctrl.abrir(_solicitud(seed=0))
    ctrl.interrumpir(sesion, n_trials_finalizados=1)
    snapshot = sesion.manifiesto
    with pytest.raises(IncompatibilidadCorridaError) as captured:
        ctrl.abrir(_solicitud(seed=1))
    assert "seed" in captured.value.motivos
    recargado = AlmacenManifiestosFs(tmp_path).cargar("run-001")
    assert recargado == snapshot


def test_completada_no_reabre(tmp_path: Path) -> None:
    ctrl, puerto = _controlador(tmp_path)
    sesion = ctrl.abrir(_solicitud())
    ctrl.completar(sesion, n_trials_finalizados=10)
    creaciones = puerto.creaciones
    otra = ctrl.abrir(_solicitud())
    assert otra.manifiesto.estado is EstadoCorrida.COMPLETADA
    assert puerto.creaciones == creaciones
    assert puerto.restauraciones == 1


def test_fallida_no_se_reabre(tmp_path: Path) -> None:
    ctrl, _ = _controlador(tmp_path)
    sesion = ctrl.abrir(_solicitud())
    ctrl.fallar(sesion, n_trials_finalizados=1)
    with pytest.raises(CorridaFallidaError):
        ctrl.abrir(_solicitud())


def test_checkpoint_idempotente(tmp_path: Path) -> None:
    ctrl, _ = _controlador(tmp_path)
    sesion = ctrl.abrir(_solicitud())
    ctrl.checkpoint(sesion, n_trials_finalizados=1)
    ctrl.checkpoint(sesion, n_trials_finalizados=1)
    assert sesion.manifiesto.estado is EstadoCorrida.EN_PROGRESO
    assert sesion.manifiesto.n_trials_finalizados == 1
