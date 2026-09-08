"""Componente 5: Asynchronous Successive Halving (ASHA) como pruner de Optuna.

Decide 'seguir o descartar'. NO recorre ventanas por su cuenta: eso es
responsabilidad exclusiva del componente 8.2 (`EjecutorGreedy`). Asincronia:
cada trial se compara contra los demas que alcanzaron el MISMO escalon, sin
esperar a que una ronda completa termine (regla de seguridad #1 y #2).

Implementado como `optuna.pruners.BasePruner` (ADR-02-006): Optuna no trae
esta regla de fabrica (sus pruners nativos no exigen un piso de ventanas de
gracia ni comparan agregados recortados por escalon), asi que se porta tal
cual desde la implementacion propia anterior, solo que ahora consulta
`study.get_trials()`/`trial.intermediate_values` en vez de un registro
interno propio.
"""

from __future__ import annotations

import optuna

from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda


class PodadorASHA(optuna.pruners.BasePruner):
    def __init__(self, reglas: ReglasPoda, *, n_ventanas_totales: int) -> None:
        if n_ventanas_totales < reglas.min_ventanas:
            raise ValueError(
                "n_ventanas_totales debe ser >= reglas.min_ventanas "
                f"({n_ventanas_totales} < {reglas.min_ventanas})"
            )
        self._reglas = reglas
        self._n_ventanas_totales = n_ventanas_totales
        self._escalones = self._calcular_escalones()
        self.ultimo_motivo: str | None = None

    def escalones(self) -> tuple[int, ...]:
        return self._escalones

    def _calcular_escalones(self) -> tuple[int, ...]:
        escalones = [self._reglas.min_ventanas]
        while escalones[-1] * self._reglas.factor_reduccion < self._n_ventanas_totales:
            escalones.append(escalones[-1] * self._reglas.factor_reduccion)
        if escalones[-1] != self._n_ventanas_totales:
            escalones.append(self._n_ventanas_totales)
        return tuple(escalones)

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        self.ultimo_motivo = None
        valores_intermedios = trial.intermediate_values
        if not valores_intermedios:
            return False

        n_evaluadas = max(valores_intermedios.keys())
        if n_evaluadas < self._reglas.min_ventanas:
            return False

        escalon = max((e for e in self._escalones if e <= n_evaluadas), default=None)
        if escalon is None:
            return False

        pares_en_escalon = {
            t.number: t.intermediate_values[escalon]
            for t in study.get_trials(deepcopy=False)
            if escalon in t.intermediate_values
        }
        if trial.number not in pares_en_escalon or len(pares_en_escalon) < 2:
            return False

        valores = sorted(pares_en_escalon.values())
        corte = max(1, len(valores) // self._reglas.factor_reduccion)
        umbral = valores[corte - 1]
        valor_actual = pares_en_escalon[trial.number]
        if valor_actual > umbral:
            self.ultimo_motivo = (
                f"escalon={escalon} valor={valor_actual:.6g} "
                f"umbral_top_{corte}={umbral:.6g}"
            )
            return True
        return False
