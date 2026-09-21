"""Unico modulo del motor de HPO que importa `optuna`.

Traduce entre los contratos de `contratos.py` / la logica pura de
`DecisorASHA` (`asha.py`) y las clases concretas de Optuna. Si el dia de
manana se cambia de backend (Ray Tune, Hyperopt, una implementacion
propia), se escribe un `adaptador_<backend>.py` con esta misma forma y
`estudio.py`/`registro.py` no cambian una linea -- solo importan de
`contratos.py`.

`volcar_jsonl`/`reanudar_estudio` viven aqui (y no en `registro.py`) por el
mismo motivo: son formato-Optuna por naturaleza (serializan/reconstruyen un
`optuna.trial.FrozenTrial`), no vocabulario propio del motor de HPO.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import optuna
from optuna.trial import TrialState, create_trial

from pred_engine.comun.logger import get_logger
from pred_engine.optimizacion.optimizadores.HPO.asha import DecisionPoda, DecisorASHA
from pred_engine.optimizacion.optimizadores.HPO.contratos import (
    EstadoTrialBackend,
    EstudioHPO,
    InfoTrial,
    TrialHPO,
)
from pred_engine.optimizacion.optimizadores.HPO.espacio import (
    Categorico,
    Entero,
    EspacioBusqueda,
    Flotante,
    Ordinal,
)

_ESTADO_A_OPTUNA: dict[EstadoTrialBackend, TrialState] = {
    "completado": TrialState.COMPLETE,
    "podado": TrialState.PRUNED,
    "fallido": TrialState.FAIL,
}
_OPTUNA_A_ESTADO: dict[TrialState, EstadoTrialBackend] = {
    v: k for k, v in _ESTADO_A_OPTUNA.items()
}

_logger = get_logger(__name__)


class PodadorASHAOptuna(optuna.pruners.BasePruner):
    """Adapta `DecisorASHA` (puro) a `optuna.pruners.BasePruner`.

    El motivo de poda de cada trial se guarda indexado por `trial.number`
    (no en un unico atributo de instancia): un pruner es una sola instancia
    compartida por todos los trials del estudio, y un atributo escalar se
    pisaria entre trials si `prune()` llegara a invocarse de forma
    concurrente (p.ej. `study.optimize(n_jobs>1)`). Con el bucle secuencial
    actual esto no se manifiesta, pero es correcto por construccion en vez
    de serlo por casualidad del orden de ejecucion.
    """

    def __init__(self, decisor: DecisorASHA) -> None:
        self._decisor = decisor
        self._motivos: dict[int, str] = {}

    def escalones(self) -> tuple[int, ...]:
        return self._decisor.escalones()

    def motivo_de(self, trial_number: int) -> str | None:
        """Consume (pop) el motivo de poda de un trial concreto."""
        return self._motivos.pop(trial_number, None)

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        valores_intermedios = trial.intermediate_values
        if not valores_intermedios:
            return False

        n_evaluadas = max(valores_intermedios.keys())
        escalon_candidatos = [e for e in self._decisor.escalones() if e <= n_evaluadas]
        if not escalon_candidatos:
            return False
        escalon = max(escalon_candidatos)

        competidores = {
            t.number: t.intermediate_values[escalon]
            for t in study.get_trials(deepcopy=False)
            if escalon in t.intermediate_values
        }

        decision: DecisionPoda = self._decisor.decidir(
            n_evaluadas=n_evaluadas,
            numero_trial=trial.number,
            valores_competidores_en_escalon=competidores,
        )
        if decision.podar and decision.motivo:
            self._motivos[trial.number] = decision.motivo
        return decision.podar


class _EstudioOptuna:
    """Implementa `EstudioHPO` sobre un `optuna.study.Study`."""

    def __init__(self, study: optuna.study.Study) -> None:
        self._study = study

    def ask(self) -> TrialHPO:
        return self._study.ask()  # optuna.trial.Trial ya satisface TrialHPO

    def tell(
        self, trial: TrialHPO, valor: float | None, *, estado: EstadoTrialBackend
    ) -> None:
        # `trial` llega tipado como `TrialHPO` (el contrato) pero en este
        # adaptador SIEMPRE es, en runtime, el `optuna.trial.Trial` que
        # devolvio `self.ask()` -- el cast es seguro, y necesario porque
        # `Study.tell()` de Optuna exige el tipo concreto.
        trial_optuna = cast(optuna.trial.Trial, trial)
        estado_optuna = _ESTADO_A_OPTUNA[estado]
        if estado == "completado":
            if valor is None:
                raise ValueError("un trial 'completado' requiere un valor")
            self._study.tell(trial_optuna, valor, state=estado_optuna)
        else:
            # Optuna prohibe pasar un valor explicito junto con PRUNED/FAIL
            # (ValueError). Para PRUNED, Optuna ya recupera automaticamente
            # el ultimo `trial.report()` como valor del trial -- por eso
            # `_correr_trial` siempre reporta antes de pedir el cierre, y
            # este adaptador no necesita (ni puede) pasarlo de nuevo aqui.
            self._study.tell(trial_optuna, state=estado_optuna)

    def trials_finalizados(self) -> list[InfoTrial]:
        salida: list[InfoTrial] = []
        for t in self._study.get_trials(deepcopy=False):
            estado = _OPTUNA_A_ESTADO.get(t.state)
            if estado is None:
                continue
            salida.append(
                InfoTrial(
                    numero=t.number,
                    estado=estado,
                    valor=t.value,
                    parametros=dict(t.params),
                    atributos=dict(t.user_attrs),
                )
            )
        return salida

    def agregar_trials_historicos(
        self, historico: Sequence[InfoTrial], *, espacio: EspacioBusqueda
    ) -> int:
        semillas = semillas_desde_historico(historico, espacio=espacio)
        return inyectar_historico(self._study, semillas)


def semillas_desde_historico(
    historico: Sequence[InfoTrial], *, espacio: EspacioBusqueda
) -> list[optuna.trial.FrozenTrial]:
    """Traduce un historico EXTERNO (`InfoTrial` de otra corrida) a
    `FrozenTrial` de Optuna, listos para `inyectar_historico` -- warm start
    (TASK-HPO-4.0-A2). Distinto de `reanudar_estudio`: ese reconstruye la
    MISMA corrida interrumpida desde su propio `backend.jsonl`; esto importa
    evidencia de OTRA corrida (ej. un segmento previo del mismo SKU) para que
    TPE arranque informado en vez de con `n_arranque` trials aleatorios --
    util quando la corrida actual tiene pocas ventanas informativas (SKU
    lumpy/disperso).

    Un trial del historico se descarta (no aborta el resto) si sus
    parametros no calzan con `espacio`: el espacio de busqueda pudo cambiar
    entre la corrida origen y la actual."""
    nombres_validos = {p.nombre for p in espacio.parametros}
    distribuciones = {p.nombre: distribucion_de(p) for p in espacio.parametros}

    semillas: list[optuna.trial.FrozenTrial] = []
    for info in historico:
        if info.estado not in _ESTADO_A_OPTUNA:
            continue
        if not set(info.parametros).issubset(nombres_validos):
            continue
        estado = _ESTADO_A_OPTUNA[info.estado]
        try:
            semilla = create_trial(
                state=estado,
                value=info.valor if estado != TrialState.FAIL else None,
                params=dict(info.parametros),
                distributions={
                    nombre: distribuciones[nombre] for nombre in info.parametros
                },
                user_attrs=dict(info.atributos) | {"warm_start_origen": True},
            )
        except ValueError:
            # Parametro presente pero con un valor fuera del rango vigente
            # del espacio actual (ej. `Entero` cuyo `alto` se redujo).
            continue
        semillas.append(semilla)
    return semillas


def inyectar_historico(
    study: optuna.study.Study, semillas: Sequence[optuna.trial.FrozenTrial]
) -> int:
    """Agrega `semillas` (de `semillas_desde_historico`) a `study` como
    trials ya evaluados. Retorna cuantas se inyectaron."""
    for semilla in semillas:
        study.add_trial(semilla)
    return len(semillas)


def persistir_estudio_hpo(estudio: EstudioHPO, ruta: str | Path) -> Path:
    if not isinstance(estudio, _EstudioOptuna):
        raise TypeError("persistencia Optuna requiere el adaptador Optuna")
    destino = Path(ruta)
    volcar_jsonl(estudio._study, destino)
    return destino


def restaurar_estudio_hpo(
    ruta: str | Path,
    *,
    espacio: EspacioBusqueda,
    muestreador: optuna.samplers.BaseSampler,
    decisor_asha: DecisorASHA,
) -> tuple[EstudioHPO, PodadorASHAOptuna]:
    podador = PodadorASHAOptuna(decisor_asha)
    study = reanudar_estudio(ruta, espacio=espacio, sampler=muestreador, pruner=podador)
    return _EstudioOptuna(study), podador


class AdaptadorPersistenciaOptuna:
    """Implementa `PuertoPersistenciaMotor` sin exponer `optuna.Study`."""

    def __init__(
        self,
        *,
        ruta_checkpoint: str | Path,
        espacio: EspacioBusqueda,
        muestreador: optuna.samplers.BaseSampler,
        decisor_asha: DecisorASHA,
    ) -> None:
        self._ruta = Path(ruta_checkpoint)
        self._espacio = espacio
        self._muestreador = muestreador
        self._decisor = decisor_asha

    def crear(self) -> object:
        return crear_estudio_optuna(
            muestreador=self._muestreador, decisor_asha=self._decisor
        )

    def persistir(self, handle: object) -> str:
        estudio, _podador = handle  # type: ignore[misc]
        persistir_estudio_hpo(estudio, self._ruta)
        return self._ruta.name

    def restaurar(self, referencia: str) -> object:
        ruta = (
            self._ruta if not referencia else self._ruta.parent / Path(referencia).name
        )
        return restaurar_estudio_hpo(
            ruta,
            espacio=self._espacio,
            muestreador=self._muestreador,
            decisor_asha=self._decisor,
        )


def crear_estudio_optuna(
    *,
    muestreador: optuna.samplers.BaseSampler,
    decisor_asha: DecisorASHA,
) -> tuple[EstudioHPO, PodadorASHAOptuna]:
    """Punto de entrada del adaptador: construye un `EstudioHPO` sobre un
    `optuna.study.Study` nuevo. Es la UNICA funcion de este archivo que
    `estudio.py` necesita llamar."""
    podador = PodadorASHAOptuna(decisor_asha)
    study = optuna.create_study(
        direction="minimize", sampler=muestreador, pruner=podador
    )
    return _EstudioOptuna(study), podador


def volcar_jsonl(study: optuna.study.Study, ruta: str | Path) -> None:
    """Serializa solo trials FINALIZADOS (COMPLETE / PRUNED / FAIL).

    Un trial RUNNING se omite a proposito: es el ensayo activo durante una
    interrupcion y debe reejecutarse desde cero al recuperar (seccion F).
    """
    from pred_engine.optimizacion.control_reanudacion.almacenamiento import (
        escribir_atomico,
    )

    finalizados = (
        TrialState.COMPLETE,
        TrialState.PRUNED,
        TrialState.FAIL,
    )
    lineas = [
        json.dumps(_frozen_trial_a_dict(t))
        for t in study.get_trials(deepcopy=False)
        if t.state in finalizados
    ]
    escribir_atomico(Path(ruta), "\n".join(lineas) + ("\n" if lineas else ""))


def _sincronizar_sampler_aleatorio(
    sampler: optuna.samplers.BaseSampler,
    espacio: EspacioBusqueda,
    n_trials_cargados: int,
) -> None:
    """Avanza el RNG del RandomSampler para alinear la reanudacion.

    `add_trial` restaura trials finalizados pero no consume el estado del
    muestreador; sin este avance, el siguiente `ask()` repite la primera
    muestra de la secuencia.

    Solo cubre `RandomSampler` (limitacion conocida, ver ADR-012). Con
    `TPESampler` el RNG interno de su fase de arranque en frio SI se
    reinicia al reanudar -- confirmado empiricamente: produce trials
    duplicados de configuraciones ya evaluadas antes de la interrupcion. No
    se replica aqui el mismo truco (auxiliar + `tell(..., FAIL)`) porque
    para TPE el conteo de muestras del RNG por `ask()` no es necesariamente
    constante (depende de cuantos trials 'buenos'/'malos' ve al decidir);
    un replay que asuma que si lo es podria dar una falsa sensacion de
    reproducibilidad sin garantizarla. Se advierte en su lugar para que la
    perdida de presupuesto de trials sea visible, no silenciosa.
    """
    if n_trials_cargados <= 0:
        return
    if not isinstance(sampler, optuna.samplers.RandomSampler):
        if isinstance(sampler, optuna.samplers.TPESampler):
            _logger.warning(
                "Reanudando con TPESampler: %d trials restaurados desde "
                "checkpoint, pero el RNG de arranque en frio del sampler no "
                "se resincroniza (limitacion conocida, ver ADR-012). Las "
                "primeras propuestas de esta corrida pueden repetir "
                "configuraciones ya evaluadas antes de la interrupcion.",
                n_trials_cargados,
            )
        return

    auxiliar = optuna.create_study(direction="minimize", sampler=sampler)
    for _ in range(n_trials_cargados):
        trial = auxiliar.ask()
        for parametro in espacio.parametros:
            if isinstance(parametro, Entero):
                trial.suggest_int(
                    parametro.nombre,
                    parametro.bajo,
                    parametro.alto,
                    step=parametro.paso,
                )
            elif isinstance(parametro, Flotante):
                trial.suggest_float(
                    parametro.nombre,
                    parametro.bajo,
                    parametro.alto,
                    log=parametro.log,
                )
            elif isinstance(parametro, Categorico):
                trial.suggest_categorical(parametro.nombre, parametro.opciones)
            elif isinstance(parametro, Ordinal):
                trial.suggest_int(parametro.nombre, 0, len(parametro.niveles) - 1)
            else:
                raise TypeError(f"tipo de parametro no soportado: {type(parametro)!r}")
        auxiliar.tell(trial, state=TrialState.FAIL)


def reanudar_estudio(
    ruta: str | Path,
    *,
    espacio: EspacioBusqueda,
    sampler: optuna.samplers.BaseSampler,
    pruner: optuna.pruners.BasePruner,
) -> optuna.study.Study:
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    ruta = Path(ruta)
    if not ruta.exists():
        return study

    distribuciones = {p.nombre: distribucion_de(p) for p in espacio.parametros}
    n_trials_cargados = 0
    with ruta.open("r", encoding="utf-8") as archivo:
        for linea in archivo:
            linea = linea.strip()
            if not linea:
                continue
            datos = json.loads(linea)
            estado = TrialState[datos["estado_optuna"]]
            trial = create_trial(
                state=estado,
                value=datos["valor"] if estado != TrialState.FAIL else None,
                params=datos["params"],
                distributions={
                    nombre: distribuciones[nombre] for nombre in datos["params"]
                },
                user_attrs=datos["user_attrs"],
                intermediate_values={
                    int(k): v for k, v in datos.get("intermediate_values", {}).items()
                },
            )
            study.add_trial(trial)
            n_trials_cargados += 1
    _sincronizar_sampler_aleatorio(sampler, espacio, n_trials_cargados)
    return study


def distribucion_de(
    parametro: Entero | Flotante | Categorico | Ordinal,
) -> optuna.distributions.BaseDistribution:
    if isinstance(parametro, Entero):
        return optuna.distributions.IntDistribution(
            parametro.bajo, parametro.alto, step=parametro.paso
        )
    if isinstance(parametro, Flotante):
        return optuna.distributions.FloatDistribution(
            parametro.bajo, parametro.alto, log=parametro.log
        )
    if isinstance(parametro, Categorico):
        return optuna.distributions.CategoricalDistribution(parametro.opciones)
    if isinstance(parametro, Ordinal):
        return optuna.distributions.IntDistribution(0, len(parametro.niveles) - 1)
    raise TypeError(f"tipo de parametro no soportado: {type(parametro)!r}")


def _frozen_trial_a_dict(t: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return {
        "estado_optuna": t.state.name,
        "params": dict(t.params),
        "valor": t.value,
        "user_attrs": dict(t.user_attrs),
        "intermediate_values": {str(k): v for k, v in t.intermediate_values.items()},
    }


def trials_desde_study(study: optuna.study.Study) -> Sequence[InfoTrial]:
    """Atajo para pruebas/scripts que ya tienen un `optuna.study.Study` y
    quieren `Sequence[InfoTrial]` sin instanciar `_EstudioOptuna` a mano."""
    return _EstudioOptuna(study).trials_finalizados()
