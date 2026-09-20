"""Orquestador del ciclo de vida de una corrida PRED."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pred_engine.comun.logger import get_logger
from pred_engine.optimizacion.control_reanudacion.contratos import (
    SCHEMA_VERSION,
    AlmacenManifiestos,
    EstadoCorrida,
    ManifiestoCorrida,
    PuertoPersistenciaMotor,
    SolicitudCorrida,
)
from pred_engine.optimizacion.control_reanudacion.errores import CorridaFallidaError
from pred_engine.optimizacion.control_reanudacion.huella import (
    calcular_huella,
    verificar_compatibilidad,
)
from pred_engine.optimizacion.control_reanudacion.transiciones import transicionar

_logger = get_logger(__name__)


@dataclass
class SesionCorrida:
    """Estado vivo de una corrida: manifiesto + handle opaco del motor."""

    manifiesto: ManifiestoCorrida
    handle: Any


class ControladorReanudacion:
    def __init__(
        self, almacen: AlmacenManifiestos, puerto: PuertoPersistenciaMotor
    ) -> None:
        self._almacen = almacen
        self._puerto = puerto

    def abrir(self, solicitud: SolicitudCorrida) -> SesionCorrida:
        if not self._almacen.existe(solicitud.run_id):
            return self._crear(solicitud)

        previo = self._almacen.cargar(solicitud.run_id)
        verificar_compatibilidad(solicitud, previo)

        if previo.estado is EstadoCorrida.FALLIDA:
            _logger.error("Rechazo: corrida %s esta fallida", solicitud.run_id)
            raise CorridaFallidaError(
                f"la corrida {solicitud.run_id} finalizo en estado fallida"
            )

        if previo.estado is EstadoCorrida.COMPLETADA:
            _logger.info(
                "Corrida %s completada; no se reabre el estudio", solicitud.run_id
            )
            handle = self._restaurar(previo)
            return SesionCorrida(manifiesto=previo, handle=handle)

        if previo.estado is EstadoCorrida.INTERRUMPIDA:
            nuevo_estado = transicionar(previo.estado, EstadoCorrida.EN_PROGRESO)
            _logger.info("Recuperacion senalizada run_id=%s", solicitud.run_id)
        elif previo.estado is EstadoCorrida.EN_PROGRESO:
            nuevo_estado = EstadoCorrida.EN_PROGRESO
            _logger.info(
                "Recuperacion no senalizada (en_progreso) run_id=%s",
                solicitud.run_id,
            )
        elif previo.estado is EstadoCorrida.NUEVA:
            nuevo_estado = transicionar(previo.estado, EstadoCorrida.EN_PROGRESO)
        else:
            nuevo_estado = previo.estado

        handle = self._restaurar(previo)
        actualizado = previo.model_copy(update={"estado": nuevo_estado})
        self._almacen.guardar(actualizado)
        return SesionCorrida(manifiesto=actualizado, handle=handle)

    def checkpoint(self, sesion: SesionCorrida, *, n_trials_finalizados: int) -> None:
        referencia = self._puerto.persistir(sesion.handle)
        estado = transicionar(sesion.manifiesto.estado, EstadoCorrida.EN_PROGRESO)
        actualizado = sesion.manifiesto.model_copy(
            update={
                "estado": estado,
                "backend_checkpoint": referencia,
                "n_trials_finalizados": n_trials_finalizados,
            }
        )
        self._almacen.guardar(actualizado)
        sesion.manifiesto = actualizado
        _logger.info(
            "Checkpoint run_id=%s trials=%d ref=%s",
            actualizado.run_id,
            n_trials_finalizados,
            referencia,
        )

    def interrumpir(self, sesion: SesionCorrida, *, n_trials_finalizados: int) -> None:
        referencia = self._puerto.persistir(sesion.handle)
        estado = transicionar(sesion.manifiesto.estado, EstadoCorrida.INTERRUMPIDA)
        actualizado = sesion.manifiesto.model_copy(
            update={
                "estado": estado,
                "backend_checkpoint": referencia,
                "n_trials_finalizados": n_trials_finalizados,
            }
        )
        self._almacen.guardar(actualizado)
        sesion.manifiesto = actualizado
        _logger.warning(
            "Interrupcion persistida run_id=%s trials=%d",
            actualizado.run_id,
            n_trials_finalizados,
        )

    def completar(self, sesion: SesionCorrida, *, n_trials_finalizados: int) -> None:
        referencia = self._puerto.persistir(sesion.handle)
        estado = transicionar(sesion.manifiesto.estado, EstadoCorrida.COMPLETADA)
        actualizado = sesion.manifiesto.model_copy(
            update={
                "estado": estado,
                "backend_checkpoint": referencia,
                "n_trials_finalizados": n_trials_finalizados,
            }
        )
        self._almacen.guardar(actualizado)
        sesion.manifiesto = actualizado
        _logger.info("Corrida completada run_id=%s", actualizado.run_id)

    def fallar(self, sesion: SesionCorrida, *, n_trials_finalizados: int) -> None:
        referencia = self._puerto.persistir(sesion.handle)
        estado = transicionar(sesion.manifiesto.estado, EstadoCorrida.FALLIDA)
        actualizado = sesion.manifiesto.model_copy(
            update={
                "estado": estado,
                "backend_checkpoint": referencia,
                "n_trials_finalizados": n_trials_finalizados,
            }
        )
        self._almacen.guardar(actualizado)
        sesion.manifiesto = actualizado
        _logger.error("Corrida fallida run_id=%s", actualizado.run_id)

    def _crear(self, solicitud: SolicitudCorrida) -> SesionCorrida:
        huella = calcular_huella(solicitud)
        manifiesto = ManifiestoCorrida(
            schema_version=SCHEMA_VERSION,
            run_id=solicitud.run_id,
            estado=EstadoCorrida.NUEVA,
            familia=solicitud.familia,
            sku_id=solicitud.sku_id,
            seed=solicitud.seed,
            metrica_objetivo=solicitud.metrica_objetivo,
            configuracion_validacion=solicitud.validacion,
            fingerprint_configuracion=huella,
            backend=solicitud.backend,
            backend_checkpoint=None,
            n_trials_objetivo=solicitud.optimizador.n_trials_objetivo,
            n_trials_finalizados=0,
        )
        self._almacen.guardar(manifiesto)
        handle = self._puerto.crear()
        referencia = self._puerto.persistir(handle)
        en_progreso = manifiesto.model_copy(
            update={
                "estado": transicionar(manifiesto.estado, EstadoCorrida.EN_PROGRESO),
                "backend_checkpoint": referencia,
            }
        )
        self._almacen.guardar(en_progreso)
        _logger.info("Corrida creada run_id=%s", solicitud.run_id)
        return SesionCorrida(manifiesto=en_progreso, handle=handle)

    def _restaurar(self, manifiesto: ManifiestoCorrida) -> Any:
        referencia = manifiesto.backend_checkpoint or ""
        return self._puerto.restaurar(referencia)
