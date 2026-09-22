"""Pronosticador Chronos-2 zero-shot para el contrato `Pronosticador`.

`fit` NO entrena: guarda el contexto. Los pesos vienen preentrenados y no se
modifican (ADR-015). Asi el Modulo 3 puede enchufar este modelo al mismo
Walk-Forward que las demas familias sin que `comun.walkforward` sepa que
existe.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from pred_engine.comun.modelos.modelos_fundacionales.configuracion import (
    CHRONOS2_ZERO_SHOT,
)
from pred_engine.comun.modelos.modelos_fundacionales.errores import (
    AjusteModeloError,
)
from pred_engine.comun.modelos.modelos_fundacionales.pipeline import (
    PipelineFundacional,
    cargar_pipeline,
)
from pred_engine.forecasting.base import BaseForecaster


class Chronos2Forecaster(BaseForecaster):
    """Siempre usa `CHRONOS2_ZERO_SHOT`: no acepta otra configuracion."""

    def __init__(
        self,
        *,
        pipeline: PipelineFundacional | None = None,
        seed: int = 0,
        forzar_no_negativo: bool = True,
    ) -> None:
        # `seed` se conserva por el contrato de `BaseForecaster`; la inferencia
        # de Chronos-2 no muestrea, asi que no la usa.
        super().__init__(seed=seed, forzar_no_negativo=forzar_no_negativo)
        self.configuracion = CHRONOS2_ZERO_SHOT
        self._pipeline = pipeline
        self._contexto: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> Chronos2Forecaster:
        serie = self._validar_serie_1d(y)
        if serie.size == 0:
            raise ValueError("y no puede estar vacio")
        if not np.all(np.isfinite(serie)):
            raise ValueError("y contiene valores no finitos")
        self._contexto = serie[-self.configuracion.max_contexto :].copy()
        self._fitted = True
        return self

    def predict_cuantiles(self, horizon: int) -> np.ndarray:
        """Cuantiles (len(cuantiles), horizon) tal como los emite el modelo.

        Unico post-proceso: el recorte a >= 0 de `BaseForecaster` (la demanda
        no puede ser negativa), igual que en SARIMA y LightGBM.
        """
        self._require_fitted()
        self._validar_horizonte(horizon)
        assert self._contexto is not None

        pipeline = self._resolver_pipeline()
        try:
            cuantiles = np.asarray(
                pipeline.pronosticar_cuantiles(self._contexto, horizon), dtype=float
            )
        except Exception as exc:
            raise AjusteModeloError(f"Chronos-2 no pronostico: {exc}") from exc

        esperado = (len(pipeline.cuantiles), horizon)
        if cuantiles.shape != esperado:
            raise AjusteModeloError(
                f"forma de cuantiles {cuantiles.shape}, se esperaba {esperado}"
            )
        return self._recortar_no_negativo(cuantiles)

    def predict(self, horizon: int) -> np.ndarray:
        cuantiles = self.predict_cuantiles(horizon)
        return cuantiles[self._indice_puntual()]

    @property
    def cuantiles(self) -> tuple[float, ...]:
        return self._resolver_pipeline().cuantiles

    def _indice_puntual(self) -> int:
        niveles = self._resolver_pipeline().cuantiles
        objetivo = self.configuracion.cuantil_puntual
        if objetivo not in niveles:
            raise AjusteModeloError(
                f"el modelo no emite el cuantil {objetivo} (emite {niveles})"
            )
        return niveles.index(objetivo)

    def _resolver_pipeline(self) -> PipelineFundacional:
        if self._pipeline is None:
            self._pipeline = cargar_pipeline(self.configuracion)
        return self._pipeline


def fabrica_fundacional(
    configuracion: Mapping[str, Any],
    *,
    seed: int,
    pipeline: PipelineFundacional | None = None,
) -> Chronos2Forecaster:
    """`FabricaPronosticador` para el Walk-Forward del Modulo 3.

    La familia fundacional no se configura: una `configuracion` no vacia es un
    intento de ajustar el modelo y se rechaza.
    """
    if configuracion:
        raise ValueError(
            "la familia fundacional no se configura ni se optimiza; se recibio "
            f"{sorted(configuracion)}"
        )
    return Chronos2Forecaster(pipeline=pipeline, seed=seed)
