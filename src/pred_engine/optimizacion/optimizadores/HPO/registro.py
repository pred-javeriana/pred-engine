"""Componente 9 (parcial): traduce el snapshot del backend al contrato externo.

La fuente de verdad de cada trial vive en el backend de HPO (hoy, un
`optuna.Study`). Este modulo NO conoce ese backend: recibe
`Sequence[InfoTrial]` -- el snapshot ya traducido por el adaptador, ver
`adaptador_optuna.py` -- y produce el contrato externo estable
(`Trial`/`ResultadoEstudio` de `comun.dataclasses.hpo`) que consumen
`classical_selection.py` y las pruebas.

Distingue 'podado' de 'fallido' con `motivo` SIEMPRE presente en ambos
(regla de seguridad #4), para poder responder despues "esta configuracion,
se descarto por poda o porque el ajuste fallo?".

La persistencia/reanudacion en JSONL (`volcar_jsonl`/`reanudar_estudio`) es
formato-Optuna por naturaleza -- serializa/reconstruye un
`optuna.trial.FrozenTrial` -- y por eso vive en `adaptador_optuna.py`, no
aqui. Este archivo no importa el backend: si algun dia aparece un `import
optuna` en estas lineas, la frontera se volvio a cruzar.
"""

from __future__ import annotations

from collections.abc import Sequence

from pred_engine.comun.dataclasses.hpo import ResultadoEstudio, Trial
from pred_engine.comun.dataclasses.validacion_temporal import VentanaTemporal
from pred_engine.optimizacion.optimizadores.HPO.contratos import InfoTrial


def instantanea_desde_estudio(
    trials_backend: Sequence[InfoTrial],
    *,
    ventanas: tuple[VentanaTemporal, ...],
    metrica_objetivo: str,
    seed: int,
    familia: str = "desconocida",
) -> ResultadoEstudio:
    trials: list[Trial] = [
        Trial(
            id=str(t.atributos.get("trial_id", t.numero)),
            configuracion=dict(t.parametros),
            estado=t.estado,
            valor=t.valor,
            n_ventanas=int(t.atributos.get("n_ventanas", 0)),
            motivo=t.atributos.get("motivo"),
            metrica_objetivo=metrica_objetivo,
            familia=str(t.atributos.get("familia", familia)),
            sku_id=t.atributos.get("sku_id"),
            timestamp=t.atributos.get("timestamp"),
        )
        for t in trials_backend
    ]

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
