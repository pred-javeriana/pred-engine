"""Componente 9 (parcial): trazabilidad y reanudacion de un estudio de HPO.

Distingue 'podado' de 'fallido' con `motivo` SIEMPRE presente en ambos
(regla de seguridad #4), para poder responder despues "esta configuracion,
se descarto por poda o porque el ajuste fallo?".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pred_engine.comun.dataclasses.hpo import ResultadoEstudio, Trial
from pred_engine.comun.dataclasses.validacion_temporal import VentanaTemporal
from pred_engine.comun.logger import get_logger

_logger = get_logger(__name__)


class RegistroEstudio:
    def __init__(
        self,
        *,
        familia: str,
        metrica_objetivo: str,
        ventanas: tuple[VentanaTemporal, ...],
        seed: int = 0,
    ) -> None:
        self._familia = familia
        self._metrica_objetivo = metrica_objetivo
        self._ventanas = ventanas
        self._seed = seed
        self._trials: dict[str, Trial] = {}

    @property
    def trials(self) -> tuple[Trial, ...]:
        return tuple(self._trials.values())

    def abrir(
        self, trial_id: str, configuracion: dict[str, Any], *, sku_id: str | None = None
    ) -> None:
        self._trials[trial_id] = Trial(
            id=trial_id,
            configuracion=dict(configuracion),
            estado="corriendo",
            valor=None,
            n_ventanas=0,
            motivo=None,
            metrica_objetivo=self._metrica_objetivo,
            familia=self._familia,
            sku_id=sku_id,
        )

    def actualizar(
        self, trial_id: str, *, n_evaluadas: int, valor_parcial: float
    ) -> None:
        actual = self._trials[trial_id]
        self._trials[trial_id] = Trial(
            id=actual.id,
            configuracion=actual.configuracion,
            estado="corriendo",
            valor=valor_parcial,
            n_ventanas=n_evaluadas,
            motivo=None,
            metrica_objetivo=actual.metrica_objetivo,
            familia=actual.familia,
            sku_id=actual.sku_id,
        )

    def completar(self, trial_id: str, *, valor: float, n_ventanas: int) -> None:
        actual = self._trials[trial_id]
        self._trials[trial_id] = Trial(
            id=actual.id,
            configuracion=actual.configuracion,
            estado="completado",
            valor=valor,
            n_ventanas=n_ventanas,
            motivo=None,
            metrica_objetivo=actual.metrica_objetivo,
            familia=actual.familia,
            sku_id=actual.sku_id,
        )
        _logger.info("Trial completado id=%s valor=%s", trial_id, valor)

    def podar(self, trial_id: str, *, ventana: int, valor: float, motivo: str) -> None:
        actual = self._trials[trial_id]
        motivo_completo = f"ventana={ventana} valor={valor:.6g} {motivo}"
        self._trials[trial_id] = Trial(
            id=actual.id,
            configuracion=actual.configuracion,
            estado="podado",
            valor=valor,
            n_ventanas=ventana,
            motivo=motivo_completo,
            metrica_objetivo=actual.metrica_objetivo,
            familia=actual.familia,
            sku_id=actual.sku_id,
        )
        _logger.info("Trial podado id=%s motivo=%s", trial_id, motivo_completo)

    def fallar(self, trial_id: str, *, motivo: str) -> None:
        actual = self._trials[trial_id]
        self._trials[trial_id] = Trial(
            id=actual.id,
            configuracion=actual.configuracion,
            estado="fallido",
            valor=None,
            n_ventanas=actual.n_ventanas,
            motivo=motivo,
            metrica_objetivo=actual.metrica_objetivo,
            familia=actual.familia,
            sku_id=actual.sku_id,
        )
        _logger.warning("Trial fallido id=%s motivo=%s", trial_id, motivo)

    def instantanea(self) -> ResultadoEstudio:
        trials = self.trials
        completados = [
            t for t in trials if t.estado == "completado" and t.valor is not None
        ]
        mejor = min(completados, key=lambda t: t.valor, default=None)  # type: ignore[arg-type]
        return ResultadoEstudio(
            mejor=mejor,
            trials=trials,
            n_completados=len(completados),
            n_podados=sum(1 for t in trials if t.estado == "podado"),
            n_fallidos=sum(1 for t in trials if t.estado == "fallido"),
            ventanas=self._ventanas,
            metrica_objetivo=self._metrica_objetivo,
            seed=self._seed,
        )

    def volcar_jsonl(self, ruta: str | Path) -> None:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with ruta.open("w", encoding="utf-8") as archivo:
            for trial in self.trials:
                archivo.write(json.dumps(_trial_a_dict(trial)) + "\n")

    @classmethod
    def reanudar(
        cls,
        ruta: str | Path,
        *,
        familia: str,
        metrica_objetivo: str,
        ventanas: tuple[VentanaTemporal, ...],
        seed: int = 0,
    ) -> RegistroEstudio:
        registro = cls(
            familia=familia,
            metrica_objetivo=metrica_objetivo,
            ventanas=ventanas,
            seed=seed,
        )
        ruta = Path(ruta)
        if not ruta.exists():
            return registro
        with ruta.open("r", encoding="utf-8") as archivo:
            for linea in archivo:
                linea = linea.strip()
                if not linea:
                    continue
                datos = json.loads(linea)
                registro._trials[datos["id"]] = Trial(**datos)
        return registro


def _trial_a_dict(trial: Trial) -> dict[str, Any]:
    return {
        "id": trial.id,
        "configuracion": dict(trial.configuracion),
        "estado": trial.estado,
        "valor": trial.valor,
        "n_ventanas": trial.n_ventanas,
        "motivo": trial.motivo,
        "metrica_objetivo": trial.metrica_objetivo,
        "familia": trial.familia,
        "sku_id": trial.sku_id,
    }
