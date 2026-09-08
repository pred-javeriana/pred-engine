"""Muestreadores de HPO: como se propone la siguiente configuracion a evaluar.

`MuestreadorAleatorio` es el baseline y el motor de pruebas de todo el
resto del sistema (ASHA, poda, registro, estudio, `classical_selection`
pueden probarse end-to-end sin el riesgo de implementacion de TPE).
`MuestreadorTPE` es la estrategia informada (Bergstra et al., 2011); ver
docs/adr/ADR-004 para la justificacion de implementarlo con NumPy en vez de
traer una dependencia externa (Optuna, etc.).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from pred_engine.comun.dataclasses.hpo import Trial
from pred_engine.optimizacion.optimizadores.HPO.espacio import (
    Categorico,
    Entero,
    EspacioBusqueda,
    Flotante,
)


class Muestreador(Protocol):
    def sugerir(self, historial: Sequence[Trial]) -> dict[str, Any]: ...


class MuestreadorAleatorio:
    def __init__(self, espacio: EspacioBusqueda, *, seed: int = 0) -> None:
        self._espacio = espacio
        self._rng = np.random.default_rng(seed)

    def sugerir(self, historial: Sequence[Trial]) -> dict[str, Any]:
        return self._espacio.muestrear(self._rng)


class MuestreadorTPE:
    def __init__(
        self,
        espacio: EspacioBusqueda,
        *,
        n_arranque: int = 10,
        gamma: float = 0.25,
        n_candidatos: int = 24,
        seed: int = 0,
    ) -> None:
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma debe estar en (0, 1)")
        self._espacio = espacio
        self._n_arranque = n_arranque
        self._gamma = gamma
        self._n_candidatos = n_candidatos
        self._rng = np.random.default_rng(seed)
        self._respaldo = MuestreadorAleatorio(espacio, seed=seed)

    def sugerir(self, historial: Sequence[Trial]) -> dict[str, Any]:
        completados = [
            t for t in historial if t.estado == "completado" and t.valor is not None
        ]
        if len(completados) < self._n_arranque:
            return self._respaldo.sugerir(historial)

        ordenados = sorted(completados, key=lambda t: t.valor)  # type: ignore[arg-type]
        n_buenas = max(1, int(round(len(ordenados) * self._gamma)))
        buenas = ordenados[:n_buenas]
        malas = ordenados[n_buenas:] or ordenados[:1]

        candidatos = [
            self._espacio.muestrear(self._rng) for _ in range(self._n_candidatos)
        ]
        mejor_candidato = candidatos[0]
        mejor_razon = -np.inf
        for candidato in candidatos:
            activos = self._espacio.nombres_activos(candidato)
            log_l = self._log_densidad(candidato, buenas, activos)
            log_g = self._log_densidad(candidato, malas, activos)
            razon = log_l - log_g
            if razon > mejor_razon:
                mejor_razon = razon
                mejor_candidato = candidato
        return mejor_candidato

    def _log_densidad(
        self,
        candidato: dict[str, Any],
        grupo: list[Trial],
        activos: tuple[str, ...],
    ) -> float:
        total = 0.0
        for parametro in self._espacio.parametros:
            if parametro.nombre not in activos:
                continue
            valores_grupo = [
                t.configuracion[parametro.nombre]
                for t in grupo
                if parametro.nombre in t.configuracion
            ]
            if not valores_grupo:
                continue
            valor_candidato = candidato[parametro.nombre]
            if isinstance(parametro, Categorico):
                total += self._log_densidad_categorica(
                    valor_candidato, valores_grupo, parametro
                )
            else:
                total += self._log_densidad_numerica(
                    valor_candidato, valores_grupo, parametro
                )
        return total

    @staticmethod
    def _log_densidad_categorica(
        valor: Any, valores_grupo: list[Any], parametro: Categorico
    ) -> float:
        n = len(valores_grupo)
        k = len(parametro.opciones)
        conteo = sum(1 for v in valores_grupo if v == valor)
        probabilidad = (conteo + 1.0) / (n + k)
        return float(np.log(probabilidad))

    def _log_densidad_numerica(
        self, valor: Any, valores_grupo: list[Any], parametro: Entero | Flotante
    ) -> float:
        usa_log = isinstance(parametro, Flotante) and parametro.log
        muestras = np.asarray(valores_grupo, dtype=float)
        if usa_log:
            muestras = np.log(muestras)
            punto = float(np.log(valor))
        else:
            punto = float(valor)

        if isinstance(parametro, Entero):
            ancho_rango = max(parametro.alto - parametro.bajo, 1)
        else:
            ancho_rango = (
                float(np.log(parametro.alto / parametro.bajo))
                if usa_log
                else parametro.alto - parametro.bajo
            )
            ancho_rango = max(ancho_rango, 1e-6)

        desviacion = float(np.std(muestras)) if len(muestras) > 1 else 0.0
        ancho_banda = max(
            1.06 * desviacion * len(muestras) ** (-1 / 5), 0.05 * ancho_rango
        )
        densidades = np.exp(-0.5 * ((punto - muestras) / ancho_banda) ** 2) / (
            ancho_banda * np.sqrt(2 * np.pi)
        )
        densidad_media = float(np.mean(densidades))
        return float(np.log(max(densidad_media, 1e-12)))
