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
)

_ESTADO_A_OPTUNA: dict[EstadoTrialBackend, TrialState] = {
    "completado": TrialState.COMPLETE,
    "podado": TrialState.PRUNED,
    "fallido": TrialState.FAIL,
}
_OPTUNA_A_ESTADO: dict[TrialState, EstadoTrialBackend] = {
    v: k for k, v in _ESTADO_A_OPTUNA.items()
}


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
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as archivo:
        for t in study.get_trials(deepcopy=False):
            archivo.write(json.dumps(_frozen_trial_a_dict(t)) + "\n")


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

    distribuciones = {p.nombre: _distribucion_de(p) for p in espacio.parametros}
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
    return study


def _distribucion_de(
    parametro: Entero | Flotante | Categorico,
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
