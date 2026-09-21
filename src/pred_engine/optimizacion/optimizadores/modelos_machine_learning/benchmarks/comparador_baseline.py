"""Seleccion por HPO vs. baselines sobre un holdout final, y ahorro de ASHA.

Protocolo: se separa el ultimo `holdout` de la serie, se selecciona sobre lo
anterior (TPE + ASHA + Walk-Forward) y se mide el MAE del pronostico a
`holdout` pasos de tres modelos entrenados con el mismo tramo: la
configuracion ganadora, LightGBM con hiperparametros por defecto y un
estacional ingenuo (`SeasonalNaiveStub`, ultimo ciclo repetido).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pred_engine.comun.dataclasses.hpo import ResultadoEstudio
from pred_engine.comun.modelos.modelos_machine_learning import fabrica_ml
from pred_engine.forecasting.base import SeasonalNaiveStub
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda
from pred_engine.optimizacion.optimizadores.modelos_machine_learning.benchmarks.series_sinteticas import (  # noqa: E501
    ClaseML,
    serie_sintetica,
)
from pred_engine.optimizacion.optimizadores.modelos_machine_learning.ml_selection import (  # noqa: E501
    seleccionar_configuracion_ml,
)

ESTACIONALIDAD = 7
# Mismas reglas que el benchmark de ASHA de 2.4 (mas agresivas que el default).
REGLAS_BENCHMARK = ReglasPoda(min_ventanas=4, factor_reduccion=2)


def ahorro_asha(estudio: ResultadoEstudio) -> float:
    """Fraccion de ventanas Walk-Forward NO evaluadas gracias a la poda.

    Proxy de costo igual al de 2.4: cada ventana es un ajuste + prediccion
    real, asi que es proporcional al tiempo sin la varianza del reloj.
    """
    evaluadas = sum(t.n_ventanas for t in estudio.trials)
    sin_poda = len(estudio.ventanas) * len(estudio.trials)
    return 1.0 - evaluadas / sin_poda


@dataclass(frozen=True, slots=True)
class ResultadoBenchmarkML:
    clase: ClaseML
    seed: int
    mae_seleccionada: float
    mae_default: float
    mae_ingenuo: float
    ahorro_asha: float

    @property
    def supera_default(self) -> bool:
        return self.mae_seleccionada <= self.mae_default

    @property
    def supera_ingenuo(self) -> bool:
        return self.mae_seleccionada <= self.mae_ingenuo


def _mae(pronostico: np.ndarray, real: np.ndarray) -> float:
    return float(np.mean(np.abs(pronostico - real)))


def comparar_con_baseline(
    clase: ClaseML,
    *,
    seed: int = 0,
    n: int = 200,
    holdout: int = 28,
    n_trials: int = 25,
) -> ResultadoBenchmarkML:
    serie = serie_sintetica(clase, n=n, seed=seed)
    entrenamiento, real = serie[:-holdout], serie[-holdout:]

    resultado = seleccionar_configuracion_ml(
        entrenamiento,
        sku_id=f"bench-{clase}-{seed}",
        n_trials=n_trials,
        horizonte=7,
        paso=7,
        seed=seed,
        reglas=REGLAS_BENCHMARK,
    )
    if resultado.seleccionada is None:
        raise RuntimeError("el benchmark no obtuvo ninguna configuracion ganadora")

    ganadora = dict(resultado.seleccionada.hiperparametros, m=ESTACIONALIDAD)
    seleccionada = fabrica_ml(ganadora, seed=seed).fit(entrenamiento)
    por_defecto = fabrica_ml({"m": ESTACIONALIDAD}, seed=seed).fit(entrenamiento)
    ingenuo = SeasonalNaiveStub(ESTACIONALIDAD, seed).fit(entrenamiento)

    return ResultadoBenchmarkML(
        clase=clase,
        seed=seed,
        mae_seleccionada=_mae(seleccionada.predict(holdout), real),
        mae_default=_mae(por_defecto.predict(holdout), real),
        mae_ingenuo=_mae(ingenuo.predict(holdout), real),
        ahorro_asha=ahorro_asha(resultado.estudio),
    )
