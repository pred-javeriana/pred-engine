"""Componente 5: Asynchronous Successive Halving (ASHA), algoritmo puro.

Decide 'seguir o descartar'. NO recorre ventanas por su cuenta: eso es
responsabilidad exclusiva del componente 8.2 (`EjecutorGreedy`). Asincronia:
cada trial se compara contra los demas que alcanzaron el MISMO escalon, sin
esperar a que una ronda completa termine (regla de seguridad #1 y #2).

`DecisorASHA` no importa ni conoce Optuna: recibe numeros (cuantas ventanas
lleva el trial, los valores de sus competidores en el mismo escalon) y
devuelve una `DecisionPoda`. Quien lo conecta a un backend concreto de HPO
es el adaptador correspondiente (hoy, `adaptador_optuna.PodadorASHAOptuna`,
que implementa `optuna.pruners.BasePruner` delegando aqui la decision).
Esta separacion es deliberada: si el dia de manana se cambia de backend,
este archivo no cambia una linea -- solo se escribe un adaptador nuevo.
"""

from __future__ import annotations

from dataclasses import dataclass

from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda


@dataclass(frozen=True, slots=True)
class DecisionPoda:
    podar: bool
    motivo: str | None = None


class DecisorASHA:
    """Algoritmo ASHA puro. No sabe que existe un `study` ni un `trial`
    concretos: opera sobre `dict[int, float]` (numero de trial -> valor)."""

    def __init__(self, reglas: ReglasPoda, *, n_ventanas_totales: int) -> None:
        if n_ventanas_totales < reglas.min_ventanas:
            raise ValueError(
                "n_ventanas_totales debe ser >= reglas.min_ventanas "
                f"({n_ventanas_totales} < {reglas.min_ventanas})"
            )
        self._reglas = reglas
        self._n_ventanas_totales = n_ventanas_totales
        self._escalones = self._calcular_escalones()

    def escalones(self) -> tuple[int, ...]:
        return self._escalones

    def _calcular_escalones(self) -> tuple[int, ...]:
        escalones = [self._reglas.min_ventanas]
        while escalones[-1] * self._reglas.factor_reduccion < self._n_ventanas_totales:
            escalones.append(escalones[-1] * self._reglas.factor_reduccion)
        if escalones[-1] != self._n_ventanas_totales:
            escalones.append(self._n_ventanas_totales)
        return tuple(escalones)

    def decidir(
        self,
        *,
        n_evaluadas: int,
        numero_trial: int,
        valores_competidores_en_escalon: dict[int, float],
    ) -> DecisionPoda:
        """Puro: recibe ya resueltos los pares (numero_trial, valor) de los
        competidores que llegaron al escalon relevante. No consulta ningun
        `study`; eso es responsabilidad del adaptador."""
        if n_evaluadas < self._reglas.min_ventanas:
            return DecisionPoda(podar=False)

        escalon = max((e for e in self._escalones if e <= n_evaluadas), default=None)
        if escalon is None:
            return DecisionPoda(podar=False)

        if (
            numero_trial not in valores_competidores_en_escalon
            or len(valores_competidores_en_escalon) < 2
        ):
            return DecisionPoda(podar=False)

        valores = sorted(valores_competidores_en_escalon.values())
        corte = max(1, len(valores) // self._reglas.factor_reduccion)
        umbral = valores[corte - 1]
        valor_propio = valores_competidores_en_escalon[numero_trial]
        if valor_propio > umbral:
            return DecisionPoda(
                podar=True,
                motivo=(
                    f"escalon={escalon} valor={valor_propio:.6g} "
                    f"umbral_top_{corte}={umbral:.6g}"
                ),
            )
        return DecisionPoda(podar=False)
