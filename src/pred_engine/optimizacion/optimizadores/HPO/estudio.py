"""Punto de entrada UNICO del motor de HPO .

Clasicos, ML y DL solo cambian `espacio` y `fabrica` el resto (muestreo,
poda semantica, asignacion de recursos ASHA, registro) es compartido. Las
ventanas se generan UNA VEZ y se comparten entre todos los trials, para que
sean comparables entre si.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pred_engine.comun.dataclasses.hpo import ResultadoEstudio
from pred_engine.comun.dataclasses.validacion_temporal import ResultadoWalkForward
from pred_engine.comun.logger import get_logger
from pred_engine.comun.walkforward.protocolos import FabricaPronosticador
from pred_engine.comun.walkforward.ventanas import generar_ventanas
from pred_engine.comun.walkforward.walk_forward_greedy import EjecutorGreedy
from pred_engine.optimizacion.optimizadores.HPO.asha import (
    AsignadorRecursosASHA,
    Decision,
)
from pred_engine.optimizacion.optimizadores.HPO.espacio import EspacioBusqueda
from pred_engine.optimizacion.optimizadores.HPO.muestreadores import (
    Muestreador,
    MuestreadorTPE,
)
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda, es_degenerada
from pred_engine.optimizacion.optimizadores.HPO.registro import RegistroEstudio

_logger = get_logger(__name__)


def ejecutar_estudio(
    y: np.ndarray,
    espacio: EspacioBusqueda,
    fabrica: FabricaPronosticador,
    n_trials: int = 50,
    min_train: int = 5,
    horizonte: int = 1,
    paso: int = 1,
    metrica_objetivo: str = "mase",
    estacionalidad: int = 7,
    muestreador: Muestreador | None = None,
    reglas: ReglasPoda | None = None,
    registro: RegistroEstudio | None = None,
    seed: int = 0,
    familia: str = "desconocida",
    sku_id: str | None = None,
) -> ResultadoEstudio:
    serie = np.asarray(y, dtype=float)
    if serie.ndim != 1:
        raise ValueError("y debe ser un array 1D")

    ventanas = generar_ventanas(
        len(serie),
        min_train=min_train,
        horizonte=horizonte,
        paso=paso,
    )
    reglas_efectivas = reglas or ReglasPoda()
    asignador = AsignadorRecursosASHA(
        reglas_efectivas, n_ventanas_totales=len(ventanas)
    )
    registro_efectivo = registro or RegistroEstudio(
        familia=familia, metrica_objetivo=metrica_objetivo, ventanas=ventanas, seed=seed
    )
    muestreador_efectivo = muestreador or MuestreadorTPE(espacio, seed=seed)

    for indice_trial in range(n_trials):
        trial_id = f"{familia}-{sku_id or 'panel'}-{indice_trial:04d}"
        configuracion = _sugerir_configuracion(
            muestreador_efectivo, registro_efectivo.trials
        )
        if configuracion is None:
            _logger.warning(
                "No se pudo muestrear una configuracion valida para %s", trial_id
            )
            continue

        registro_efectivo.abrir(trial_id, configuracion, sku_id=sku_id)
        ejecutor = EjecutorGreedy(
            serie,
            fabrica,
            configuracion,
            ventanas=ventanas,
            metrica_objetivo=metrica_objetivo,
            estacionalidad=estacionalidad,
            agregacion=reglas_efectivas.agregacion,
            seed=seed,
            tolerar_fallos=True,
            identificador=trial_id,
        )

        _correr_trial(
            ejecutor, asignador, registro_efectivo, trial_id, reglas_efectivas
        )

    resultado = registro_efectivo.instantanea()
    _logger.info(
        "Estudio %s completado: %d completados, %d podados, %d fallidos",
        familia,
        resultado.n_completados,
        resultado.n_podados,
        resultado.n_fallidos,
    )
    return resultado


def _sugerir_configuracion(
    muestreador: Muestreador, historial: tuple[Any, ...]
) -> dict[str, Any] | None:
    try:
        return muestreador.sugerir(historial)
    except Exception:
        _logger.exception("Fallo al sugerir configuracion")
        return None


def _correr_trial(
    ejecutor: EjecutorGreedy,
    asignador: AsignadorRecursosASHA,
    registro: RegistroEstudio,
    trial_id: str,
    reglas: ReglasPoda,
) -> None:
    while True:
        estado = ejecutor.avanzar()
        if estado is None:
            resultado = ejecutor.resultado()
            _cerrar_por_agotamiento(registro, trial_id, resultado)
            return

        registro.actualizar(
            trial_id, n_evaluadas=estado.n_evaluadas, valor_parcial=estado.valor_parcial
        )

        if estado.ultima.fallo is None and reglas.habilitar_poda_semantica:
            es_mala, motivo_semantico = es_degenerada(
                estado.ultima.y_pred, y_train=estado.ultima.y_train
            )
            if es_mala:
                motivo_completo = f"poda_semantica:{motivo_semantico}"
                ejecutor.cerrar(motivo_completo)
                registro.podar(
                    trial_id,
                    ventana=estado.n_evaluadas,
                    valor=estado.valor_parcial,
                    motivo=motivo_completo,
                )
                return

        asignador.registrar(trial_id, estado)
        decision, motivo = asignador.decidir(trial_id)
        if decision is Decision.PODAR:
            motivo_final = motivo or "asha"
            ejecutor.cerrar(motivo_final)
            _cerrar_por_decision_asha(
                registro,
                trial_id,
                estado.n_evaluadas,
                estado.valor_parcial,
                motivo_final,
            )
            return


def _cerrar_por_agotamiento(
    registro: RegistroEstudio, trial_id: str, resultado: ResultadoWalkForward
) -> None:
    if math.isfinite(resultado.valor_agregado):
        registro.completar(
            trial_id,
            valor=resultado.valor_agregado,
            n_ventanas=resultado.n_ventanas_evaluadas,
        )
    else:
        registro.fallar(trial_id, motivo="ninguna_ventana_convergio")


def _cerrar_por_decision_asha(
    registro: RegistroEstudio,
    trial_id: str,
    ventana: int,
    valor_parcial: float,
    motivo: str,
) -> None:
    if math.isfinite(valor_parcial):
        registro.podar(trial_id, ventana=ventana, valor=valor_parcial, motivo=motivo)
    else:
        registro.fallar(
            trial_id, motivo=f"ninguna_ventana_convergio_hasta_la_poda:{motivo}"
        )
