"""Punto de entrada UNICO del motor de HPO.

Clasicos, ML y DL solo cambian `espacio` y `fabrica` el resto (muestreo,
poda semantica, asignacion de recursos ASHA, registro) es compartido. Las
ventanas se generan UNA VEZ y se comparten entre todos los trials, para que
sean comparables entre si.

ADR-02-006: el bucle usa la interfaz ask/tell de Optuna -- `EjecutorGreedy`
sigue siendo quien decide cuando avanzar una ventana (ADR-02-004) y
`PodadorASHA` quien decide cuando podar (ADR-02-005); Optuna solo aporta el
muestreador (`TPESampler`) y el `Study` como registro de trials. El
contrato externo (`Trial`/`ResultadoEstudio`) no cambia.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import optuna
from optuna.trial import TrialState

from pred_engine.comun.dataclasses.hpo import ResultadoEstudio
from pred_engine.comun.dataclasses.validacion_temporal import ResultadoWalkForward
from pred_engine.comun.logger import get_logger
from pred_engine.comun.walkforward.protocolos import FabricaPronosticador
from pred_engine.comun.walkforward.ventanas import generar_ventanas
from pred_engine.comun.walkforward.walk_forward_greedy import EjecutorGreedy
from pred_engine.optimizacion.optimizadores.HPO.asha import PodadorASHA
from pred_engine.optimizacion.optimizadores.HPO.espacio import (
    Categorico,
    Entero,
    EspacioBusqueda,
    Flotante,
    Parametro,
)
from pred_engine.optimizacion.optimizadores.HPO.muestreadores import (
    construir_muestreador_tpe,
)
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda, es_degenerada
from pred_engine.optimizacion.optimizadores.HPO.registro import (
    instantanea_desde_estudio,
)

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
    muestreador: optuna.samplers.BaseSampler | None = None,
    reglas: ReglasPoda | None = None,
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
    pruner = PodadorASHA(reglas_efectivas, n_ventanas_totales=len(ventanas))
    sampler = muestreador or construir_muestreador_tpe(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    for indice_trial in range(n_trials):
        trial_id = f"{familia}-{sku_id or 'panel'}-{indice_trial:04d}"
        trial = study.ask()
        trial.set_user_attr("trial_id", trial_id)
        trial.set_user_attr("familia", familia)
        if sku_id is not None:
            trial.set_user_attr("sku_id", sku_id)

        configuracion = _sugerir_configuracion(trial, espacio)
        if configuracion is None:
            trial.set_user_attr("motivo", "configuracion_invalida")
            study.tell(trial, state=TrialState.FAIL)
            _logger.warning("Configuracion invalida (restricciones) para %s", trial_id)
            continue

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

        _correr_trial(trial, ejecutor, study, pruner, reglas_efectivas)

    resultado = instantanea_desde_estudio(
        study,
        ventanas=ventanas,
        metrica_objetivo=metrica_objetivo,
        seed=seed,
        familia=familia,
    )
    _logger.info(
        "Estudio %s completado: %d completados, %d podados, %d fallidos",
        familia,
        resultado.n_completados,
        resultado.n_podados,
        resultado.n_fallidos,
    )
    return resultado


def _sugerir_configuracion(
    trial: optuna.trial.Trial, espacio: EspacioBusqueda
) -> dict[str, Any] | None:
    condiciones = {c.parametro: c for c in espacio.condiciones}
    configuracion: dict[str, Any] = {}
    for parametro in espacio.parametros:
        condicion = condiciones.get(parametro.nombre)
        if condicion is not None and configuracion.get(condicion.padre) not in (
            condicion.valores
        ):
            continue
        configuracion[parametro.nombre] = _sugerir_parametro(trial, parametro)
    if not espacio.es_valida(configuracion):
        return None
    return configuracion


def _sugerir_parametro(trial: optuna.trial.Trial, parametro: Parametro) -> Any:
    if isinstance(parametro, Entero):
        return trial.suggest_int(
            parametro.nombre, parametro.bajo, parametro.alto, step=parametro.paso
        )
    if isinstance(parametro, Flotante):
        return trial.suggest_float(
            parametro.nombre, parametro.bajo, parametro.alto, log=parametro.log
        )
    if isinstance(parametro, Categorico):
        return trial.suggest_categorical(parametro.nombre, parametro.opciones)
    raise TypeError(f"tipo de parametro no soportado: {type(parametro)!r}")


def _correr_trial(
    trial: optuna.trial.Trial,
    ejecutor: EjecutorGreedy,
    study: optuna.study.Study,
    pruner: PodadorASHA,
    reglas: ReglasPoda,
) -> None:
    while True:
        estado = ejecutor.avanzar()
        if estado is None:
            resultado = ejecutor.resultado()
            _cerrar_por_agotamiento(trial, study, resultado)
            return

        trial.report(estado.valor_parcial, step=estado.n_evaluadas)

        if estado.ultima.fallo is None and reglas.habilitar_poda_semantica:
            es_mala, motivo_semantico = es_degenerada(
                estado.ultima.y_pred, y_train=estado.ultima.y_train
            )
            if es_mala:
                motivo_completo = (
                    f"ventana={estado.n_evaluadas} valor={estado.valor_parcial:.6g} "
                    f"poda_semantica:{motivo_semantico}"
                )
                ejecutor.cerrar(f"poda_semantica:{motivo_semantico}")
                _cerrar_podado_o_fallido(
                    trial,
                    study,
                    estado.n_evaluadas,
                    estado.valor_parcial,
                    motivo_completo,
                )
                return

        if trial.should_prune():
            motivo_completo = pruner.ultimo_motivo or (
                f"ventana={estado.n_evaluadas} valor={estado.valor_parcial:.6g} asha"
            )
            ejecutor.cerrar("asha")
            _cerrar_podado_o_fallido(
                trial,
                study,
                estado.n_evaluadas,
                estado.valor_parcial,
                motivo_completo,
            )
            return


def _cerrar_por_agotamiento(
    trial: optuna.trial.Trial,
    study: optuna.study.Study,
    resultado: ResultadoWalkForward,
) -> None:
    trial.set_user_attr("n_ventanas", resultado.n_ventanas_evaluadas)
    if math.isfinite(resultado.valor_agregado):
        study.tell(trial, resultado.valor_agregado, state=TrialState.COMPLETE)
    else:
        trial.set_user_attr("motivo", "ninguna_ventana_convergio")
        study.tell(trial, state=TrialState.FAIL)


def _cerrar_podado_o_fallido(
    trial: optuna.trial.Trial,
    study: optuna.study.Study,
    n_evaluadas: int,
    valor_parcial: float,
    motivo: str,
) -> None:
    trial.set_user_attr("n_ventanas", n_evaluadas)
    trial.set_user_attr("motivo", motivo)
    if math.isfinite(valor_parcial):
        study.tell(trial, state=TrialState.PRUNED)
    else:
        study.tell(trial, state=TrialState.FAIL)
