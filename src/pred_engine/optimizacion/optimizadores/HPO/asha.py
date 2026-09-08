"""Componente 5: Asynchronous Successive Halving (ASHA) + poda greedy.

Decide 'seguir o descartar'. NO recorre ventanas por su cuenta: eso es
responsabilidad exclusiva del componente 8.2 (`EjecutorGreedy`). Asincronia:
cada trial se compara contra los demas que alcanzaron el MISMO escalon, sin
esperar a que una ronda completa termine (regla de seguridad #1 y #2).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pred_engine.comun.dataclasses.validacion_temporal import EstadoParcial
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda


class Decision(StrEnum):
    CONTINUAR = "continuar"
    PODAR = "podar"


@dataclass
class _RegistroTrialASHA:
    ultimo_n_evaluadas: int = 0
    ultimo_valor: float = float("inf")


class AsignadorRecursosASHA:
    def __init__(self, reglas: ReglasPoda, *, n_ventanas_totales: int) -> None:
        if n_ventanas_totales < reglas.min_ventanas:
            raise ValueError(
                "n_ventanas_totales debe ser >= reglas.min_ventanas "
                f"({n_ventanas_totales} < {reglas.min_ventanas})"
            )
        self._reglas = reglas
        self._n_ventanas_totales = n_ventanas_totales
        self._escalones = self._calcular_escalones()
        self._valores_por_escalon: dict[int, dict[str, float]] = {
            e: {} for e in self._escalones
        }
        self._estado: dict[str, _RegistroTrialASHA] = {}

    def escalones(self) -> tuple[int, ...]:
        return self._escalones

    def _calcular_escalones(self) -> tuple[int, ...]:
        escalones = [self._reglas.min_ventanas]
        while escalones[-1] * self._reglas.factor_reduccion < self._n_ventanas_totales:
            escalones.append(escalones[-1] * self._reglas.factor_reduccion)
        if escalones[-1] != self._n_ventanas_totales:
            escalones.append(self._n_ventanas_totales)
        return tuple(escalones)

    def registrar(self, trial_id: str, estado: EstadoParcial) -> None:
        registro = self._estado.setdefault(trial_id, _RegistroTrialASHA())
        registro.ultimo_n_evaluadas = estado.n_evaluadas
        registro.ultimo_valor = estado.valor_parcial
        if estado.n_evaluadas in self._valores_por_escalon:
            self._valores_por_escalon[estado.n_evaluadas][trial_id] = (
                estado.valor_parcial
            )

    def decidir(self, trial_id: str) -> tuple[Decision, str | None]:
        registro = self._estado.get(trial_id)
        if registro is None:
            return Decision.CONTINUAR, None

        if registro.ultimo_n_evaluadas < self._reglas.min_ventanas:
            return Decision.CONTINUAR, None

        escalon_alcanzado = max(
            (e for e in self._escalones if e <= registro.ultimo_n_evaluadas),
            default=None,
        )
        if escalon_alcanzado is None:
            return Decision.CONTINUAR, None

        pares_en_escalon = self._valores_por_escalon[escalon_alcanzado]
        if trial_id not in pares_en_escalon or len(pares_en_escalon) < 2:
            return Decision.CONTINUAR, None

        valores = sorted(pares_en_escalon.values())
        corte = max(1, len(valores) // self._reglas.factor_reduccion)
        umbral = valores[corte - 1]
        if pares_en_escalon[trial_id] > umbral:
            return (
                Decision.PODAR,
                f"escalon={escalon_alcanzado} valor={pares_en_escalon[trial_id]:.6g} "
                f"umbral_top_{corte}={umbral:.6g}",
            )
        return Decision.CONTINUAR, None
