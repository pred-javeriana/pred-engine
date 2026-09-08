"""Contratos estructurales del motor de HPO -- frontera con el backend.

`estudio.py` y la logica de decision de ASHA (`asha.py`) hablan SOLO estos
`Protocol` y la dataclass `InfoTrial`. Ningun import de un backend concreto
(hoy Optuna) cruza esta frontera hacia el motor: si el dia de manana se
cambia Optuna por otro backend, se escribe un adaptador nuevo que
satisfaga estos contratos y ni `estudio.py` ni `asha.py` cambian una linea.

`TrialHPO` nombra sus metodos Y sus parametros exactamente igual que
`optuna.trial.Trial` (`suggest_int(name, low, high, ...)`, `report(value,
step)`...) a proposito, y no es cosmetico: son verbos estandar en HPO (no
jerga propietaria de Optuna), y en Python un `Protocol` se satisface de
forma ESTRUCTURAL -- si los nombres de parametro no coincidieran,
`optuna.trial.Trial` dejaria de cumplir este contrato para un chequeo de
tipos estatico (pyright/mypy) aunque `isinstance()` en runtime no lo note
(ese solo mira que los metodos existan, no sus firmas exactas). Ventaja
practica de mantener la coincidencia: mientras el backend sea Optuna, su
`Trial` ya satisface este `Protocol` sin necesidad de una clase envoltorio,
en runtime Y en el chequeo estatico.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

EstadoTrialBackend = Literal["completado", "podado", "fallido"]


@runtime_checkable
class TrialHPO(Protocol):
    """Lo que el motor necesita de UN trial en curso."""

    @property
    def number(self) -> int: ...

    def suggest_int(self, name: str, low: int, high: int, *, step: int = 1) -> int: ...

    def suggest_float(
        self, name: str, low: float, high: float, *, log: bool = False
    ) -> float: ...

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any: ...

    def set_user_attr(self, key: str, value: Any) -> None: ...

    def report(self, value: float, step: int) -> None: ...

    def should_prune(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class InfoTrial:
    """Snapshot de un trial ya cerrado, en vocabulario propio (no Optuna)."""

    numero: int
    estado: EstadoTrialBackend
    valor: float | None
    parametros: Mapping[str, Any]
    atributos: Mapping[str, Any]


class EstudioHPO(Protocol):
    """Lo que el motor necesita del backend para orquestar trials."""

    def ask(self) -> TrialHPO: ...

    def tell(
        self, trial: TrialHPO, valor: float | None, *, estado: EstadoTrialBackend
    ) -> None: ...

    def trials_finalizados(self) -> Sequence[InfoTrial]: ...


class ProveedorMotivoPoda(Protocol):
    """Lo minimo que `estudio.py` necesita saber de 'quien podo y por que'."""

    def motivo_de(self, trial_number: int) -> str | None: ...
