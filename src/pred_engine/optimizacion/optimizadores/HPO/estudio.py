"""Punto de entrada UNICO del motor de HPO.

Clasicos, ML y DL solo cambian `espacio` y `fabrica` el resto (muestreo,
poda semantica, asignacion de recursos ASHA, registro) es compartido. Las
ventanas se generan UNA VEZ y se comparten entre todos los trials, para que
sean comparables entre si.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from pred_engine.comun.dataclasses.hpo import ResultadoEstudio
from pred_engine.comun.dataclasses.validacion_temporal import ResultadoWalkForward
from pred_engine.comun.logger import get_logger
from pred_engine.comun.walkforward.protocolos import FabricaPronosticador
from pred_engine.comun.walkforward.ventanas import generar_ventanas
from pred_engine.comun.walkforward.walk_forward_greedy import EjecutorGreedy
from pred_engine.optimizacion.control_reanudacion import (
    AlmacenManifiestosFs,
    ConfiguracionOptimizador,
    ConfiguracionValidacion,
    ControladorReanudacion,
    EstadoCorrida,
    SolicitudCorrida,
)
from pred_engine.optimizacion.optimizadores.HPO.adaptador_optuna import (
    AdaptadorPersistenciaOptuna,
    crear_estudio_optuna,
)
from pred_engine.optimizacion.optimizadores.HPO.asha import DecisorASHA
from pred_engine.optimizacion.optimizadores.HPO.contratos import (
    EstudioHPO,
    InfoTrial,
    ProveedorMotivoPoda,
    TrialHPO,
)
from pred_engine.optimizacion.optimizadores.HPO.errores import EstudioError
from pred_engine.optimizacion.optimizadores.HPO.espacio import (
    Categorico,
    Entero,
    EspacioBusqueda,
    Flotante,
    Ordinal,
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

# Tope de intentos de muestreo invalido (restricciones de EspacioBusqueda)
# QUE NO CUENTAN contra `n_trials`, por cada trial real que si se necesita
# evaluar. Evita que un espacio con restricciones agresivas degrade el
# presupuesto de busqueda en silencio (varios `n_trials` gastados en
# configuraciones nunca evaluadas) y, a la vez, evita un bucle infinito si
# el espacio rechaza casi todo.
_MAX_INTENTOS_INVALIDOS_POR_TRIAL = 20


def huella_serie(y: np.ndarray) -> str:
    arr = np.asarray(y, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _etiqueta_muestreador(muestreador: Any | None) -> str:
    if muestreador is None:
        return "tpe"
    return type(muestreador).__name__


def _cablear_controlador(
    *,
    raiz_corrida: str | Path,
    run_id: str,
    espacio: EspacioBusqueda,
    muestreador: Any,
    decisor: DecisorASHA,
) -> ControladorReanudacion:
    dir_run = Path(raiz_corrida) / run_id
    dir_run.mkdir(parents=True, exist_ok=True)
    puerto = AdaptadorPersistenciaOptuna(
        ruta_checkpoint=dir_run / "backend.jsonl",
        espacio=espacio,
        muestreador=muestreador,
        decisor_asha=decisor,
    )
    return ControladorReanudacion(AlmacenManifiestosFs(raiz_corrida), puerto)


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
    muestreador: Any | None = None,
    reglas: ReglasPoda | None = None,
    seed: int = 0,
    familia: str = "desconocida",
    sku_id: str | None = None,
    controlador: ControladorReanudacion | None = None,
    raiz_corrida: str | Path | None = None,
    run_id: str | None = None,
    historico_previo: Sequence[InfoTrial] | None = None,
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
    decisor = DecisorASHA(reglas_efectivas, n_ventanas_totales=len(ventanas))
    muestreador_efectivo = muestreador or construir_muestreador_tpe(seed=seed)

    controlador_efectivo = controlador
    if controlador_efectivo is None and raiz_corrida is not None:
        if not run_id:
            raise EstudioError("run_id es obligatorio cuando se indica raiz_corrida")
        controlador_efectivo = _cablear_controlador(
            raiz_corrida=raiz_corrida,
            run_id=run_id,
            espacio=espacio,
            muestreador=muestreador_efectivo,
            decisor=decisor,
        )
    elif controlador_efectivo is not None and not run_id:
        raise EstudioError("run_id es obligatorio cuando hay controlador")

    sesion = None
    if controlador_efectivo is not None:
        assert run_id is not None
        solicitud = SolicitudCorrida(
            run_id=run_id,
            familia=familia,
            sku_id=sku_id or "panel",
            seed=seed,
            metrica_objetivo=metrica_objetivo,
            validacion=ConfiguracionValidacion(
                min_train=min_train,
                horizonte=horizonte,
                paso=paso,
                estacionalidad=estacionalidad,
            ),
            espacio_busqueda=espacio.descripcion_canonica(),
            optimizador=ConfiguracionOptimizador(
                n_trials_objetivo=n_trials,
                muestreador=_etiqueta_muestreador(muestreador),
                min_ventanas=reglas_efectivas.min_ventanas,
                agregacion=reglas_efectivas.agregacion,
                proporcion_recorte=reglas_efectivas.proporcion_recorte,
                factor_reduccion=reglas_efectivas.factor_reduccion,
                habilitar_poda_semantica=reglas_efectivas.habilitar_poda_semantica,
            ),
            n_observaciones=int(serie.size),
            huella_serie=huella_serie(serie),
        )
        sesion = controlador_efectivo.abrir(solicitud)
        study, podador = sesion.handle  # type: ignore[misc]
        indice_trial = sesion.manifiesto.n_trials_finalizados
        if sesion.manifiesto.estado is EstadoCorrida.COMPLETADA:
            _logger.info(
                "Estudio %s ya completado; se reconstruye el resultado", run_id
            )
            return instantanea_desde_estudio(
                study.trials_finalizados(),
                ventanas=tuple(ventanas),
                metrica_objetivo=metrica_objetivo,
                seed=seed,
                familia=familia,
            )
    else:
        study, podador = crear_estudio_optuna(
            muestreador=muestreador_efectivo, decisor_asha=decisor
        )
        indice_trial = 0

    if historico_previo and indice_trial == 0:
        n_inyectados = study.agregar_trials_historicos(
            historico_previo, espacio=espacio
        )
        _logger.info(
            "Warm start: %d/%d trials del historico previo inyectados en %s",
            n_inyectados,
            len(historico_previo),
            familia,
        )

    intentos_invalidos = 0
    try:
        while indice_trial < n_trials:
            trial_id = f"{familia}-{sku_id or 'panel'}-{indice_trial:04d}"
            trial = study.ask()
            trial.set_user_attr("trial_id", trial_id)
            trial.set_user_attr("timestamp", datetime.now(UTC).isoformat())
            trial.set_user_attr("familia", familia)
            if sku_id is not None:
                trial.set_user_attr("sku_id", sku_id)

            configuracion = _sugerir_configuracion(trial, espacio)
            if configuracion is None:
                trial.set_user_attr("motivo", "configuracion_invalida")
                study.tell(trial, None, estado="fallido")
                _logger.warning(
                    "Configuracion invalida (restricciones) para %s", trial_id
                )
                intentos_invalidos += 1
                if sesion is not None:
                    assert controlador_efectivo is not None
                    controlador_efectivo.checkpoint(
                        sesion, n_trials_finalizados=indice_trial
                    )
                if intentos_invalidos > n_trials * _MAX_INTENTOS_INVALIDOS_POR_TRIAL:
                    raise EstudioError(
                        "el espacio de busqueda rechaza casi todas las "
                        "configuraciones muestreadas "
                        f"({intentos_invalidos} intentos invalidos); revise "
                        "las restricciones de EspacioBusqueda"
                    )
                continue
            intentos_invalidos = 0

            ejecutor = EjecutorGreedy(
                serie,
                fabrica,
                configuracion,
                ventanas=ventanas,
                metrica_objetivo=metrica_objetivo,
                estacionalidad=estacionalidad,
                agregacion=reglas_efectivas.agregacion,
                proporcion_recorte=reglas_efectivas.proporcion_recorte,
                seed=seed,
                tolerar_fallos=True,
                identificador=trial_id,
            )

            _correr_trial(trial, ejecutor, study, podador, reglas_efectivas)
            indice_trial += 1
            if sesion is not None:
                assert controlador_efectivo is not None
                controlador_efectivo.checkpoint(
                    sesion, n_trials_finalizados=indice_trial
                )

        resultado = instantanea_desde_estudio(
            study.trials_finalizados(),
            ventanas=tuple(ventanas),
            metrica_objetivo=metrica_objetivo,
            seed=seed,
            familia=familia,
        )
        if sesion is not None:
            assert controlador_efectivo is not None
            controlador_efectivo.completar(sesion, n_trials_finalizados=indice_trial)
        _logger.info(
            "Estudio %s completado: %d completados, %d podados, %d fallidos",
            familia,
            resultado.n_completados,
            resultado.n_podados,
            resultado.n_fallidos,
        )
        return resultado
    except EstudioError:
        if sesion is not None:
            assert controlador_efectivo is not None
            controlador_efectivo.fallar(sesion, n_trials_finalizados=indice_trial)
        raise
    except (Exception, KeyboardInterrupt):
        if sesion is not None:
            try:
                assert controlador_efectivo is not None
                controlador_efectivo.interrumpir(
                    sesion, n_trials_finalizados=indice_trial
                )
            except Exception:
                _logger.exception("no se pudo persistir la interrupcion de %s", run_id)
        raise


def _sugerir_configuracion(
    trial: TrialHPO, espacio: EspacioBusqueda
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


def _sugerir_parametro(trial: TrialHPO, parametro: Parametro) -> Any:
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
    if isinstance(parametro, Ordinal):
        return trial.suggest_int(parametro.nombre, 0, len(parametro.niveles) - 1)
    raise TypeError(f"tipo de parametro no soportado: {type(parametro)!r}")


def _correr_trial(
    trial: TrialHPO,
    ejecutor: EjecutorGreedy,
    study: EstudioHPO,
    pruner: ProveedorMotivoPoda,
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
            motivo_completo = pruner.motivo_de(trial.number) or (
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
    trial: TrialHPO,
    study: EstudioHPO,
    resultado: ResultadoWalkForward,
) -> None:
    trial.set_user_attr("n_ventanas", resultado.n_ventanas_evaluadas)
    if math.isfinite(resultado.valor_agregado):
        study.tell(trial, resultado.valor_agregado, estado="completado")
    else:
        trial.set_user_attr("motivo", "ninguna_ventana_convergio")
        study.tell(trial, None, estado="fallido")


def _cerrar_podado_o_fallido(
    trial: TrialHPO,
    study: EstudioHPO,
    n_evaluadas: int,
    valor_parcial: float,
    motivo: str,
) -> None:
    trial.set_user_attr("n_ventanas", n_evaluadas)
    trial.set_user_attr("motivo", motivo)
    if math.isfinite(valor_parcial):
        # No se pasa `valor_parcial` aqui: ya se reporto via `trial.report()`
        # en `_correr_trial` justo antes de esta llamada, y el backend
        # (Optuna) recupera automaticamente ese ultimo valor reportado como
        # el valor del trial podado.
        study.tell(trial, None, estado="podado")
    else:
        study.tell(trial, None, estado="fallido")
