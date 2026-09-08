"""Componente 9 (parcial): trazabilidad y reanudacion de un estudio de HPO.

ADR-02-006: la fuente de verdad de cada trial ahora es el `optuna.Study`
(via ask/tell), no una clase de registro propia. Este modulo solo traduce
entre ese `Study` y el contrato externo estable (`Trial`/`ResultadoEstudio`
de `comun.dataclasses.hpo`) que ya consumen `classical_selection.py` y las
pruebas -- y provee el volcado/reanudacion en JSONL que antes ofrecia
`RegistroEstudio`.

Distingue 'podado' de 'fallido' con `motivo` SIEMPRE presente en ambos
(regla de seguridad #4), para poder responder despues "esta configuracion,
se descarto por poda o porque el ajuste fallo?".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState, create_trial

from pred_engine.comun.dataclasses.hpo import EstadoTrial, ResultadoEstudio, Trial
from pred_engine.comun.dataclasses.validacion_temporal import VentanaTemporal
from pred_engine.optimizacion.optimizadores.HPO.espacio import (
    Categorico,
    Entero,
    EspacioBusqueda,
    Flotante,
)

_ESTADO_OPTUNA_A_TRIAL: dict[TrialState, EstadoTrial] = {
    TrialState.COMPLETE: "completado",
    TrialState.PRUNED: "podado",
    TrialState.FAIL: "fallido",
}


def instantanea_desde_estudio(
    study: optuna.study.Study,
    *,
    ventanas: tuple[VentanaTemporal, ...],
    metrica_objetivo: str,
    seed: int,
    familia: str = "desconocida",
) -> ResultadoEstudio:
    trials: list[Trial] = []
    for t in study.get_trials(deepcopy=False):
        estado = _ESTADO_OPTUNA_A_TRIAL.get(t.state)
        if estado is None:
            continue
        valor = t.value if estado in ("completado", "podado") else None
        trials.append(
            Trial(
                id=str(t.user_attrs.get("trial_id", t.number)),
                configuracion=dict(t.params),
                estado=estado,
                valor=valor,
                n_ventanas=int(t.user_attrs.get("n_ventanas", 0)),
                motivo=t.user_attrs.get("motivo"),
                metrica_objetivo=metrica_objetivo,
                familia=str(t.user_attrs.get("familia", familia)),
                sku_id=t.user_attrs.get("sku_id"),
            )
        )

    completados = [
        t for t in trials if t.estado == "completado" and t.valor is not None
    ]
    mejor = min(completados, key=lambda t: t.valor, default=None)  # type: ignore[arg-type]
    return ResultadoEstudio(
        mejor=mejor,
        trials=tuple(trials),
        n_completados=len(completados),
        n_podados=sum(1 for t in trials if t.estado == "podado"),
        n_fallidos=sum(1 for t in trials if t.estado == "fallido"),
        ventanas=ventanas,
        metrica_objetivo=metrica_objetivo,
        seed=seed,
    )


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
