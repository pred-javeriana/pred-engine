"""Carga del modelo fundacional: el UNICO modulo que toca `torch` y `chronos`.

`torch`/`chronos` se importan dentro de `cargar_pipeline`, asi que el paquete
se importa sin el extra `foundation` (CI no lo instala). El resto del codigo
habla con el modelo via `PipelineFundacional`, lo que permite probarlo con
un doble determinista en milisegundos.
"""

from __future__ import annotations

from functools import cache
from typing import Any, Protocol, runtime_checkable

import numpy as np

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos.modelos_fundacionales.configuracion import (
    ConfiguracionFundacional,
)
from pred_engine.comun.modelos.modelos_fundacionales.errores import (
    ModeloFundacionalNoDisponibleError,
)

_logger = get_logger(__name__)


@runtime_checkable
class PipelineFundacional(Protocol):
    """Pronostico por cuantiles de UNA serie univariada."""

    @property
    def cuantiles(self) -> tuple[float, ...]: ...

    def pronosticar_cuantiles(self, contexto: np.ndarray, horizonte: int) -> np.ndarray:
        """Devuelve un arreglo (len(cuantiles), horizonte)."""
        ...


class _AdaptadorChronos2:
    """Envuelve `chronos.Chronos2Pipeline` (API de chronos-forecasting 2.3)."""

    def __init__(self, pipeline: Any, config: ConfiguracionFundacional) -> None:
        self._pipeline = pipeline
        self._config = config
        self._cuantiles = tuple(float(q) for q in pipeline.quantiles)

    @property
    def cuantiles(self) -> tuple[float, ...]:
        return self._cuantiles

    def pronosticar_cuantiles(
        self, contexto: np.ndarray, horizonte: int
    ) -> np.ndarray:  # pragma: no cover - requiere el modelo real (prueba slow)
        import torch  # pyright: ignore[reportMissingImports]

        with torch.inference_mode():
            salida = self._pipeline.predict(
                [np.asarray(contexto, dtype=np.float32)],
                prediction_length=horizonte,
                context_length=self._config.max_contexto,
                cross_learning=self._config.aprendizaje_cruzado,
            )
        # Un elemento por serie, de forma (n_variates=1, n_cuantiles, horizonte).
        return salida[0][0].detach().cpu().numpy().astype(float)


@cache
def cargar_pipeline(config: ConfiguracionFundacional) -> PipelineFundacional:
    """Carga el modelo UNA vez por proceso y configuracion, con pesos congelados."""
    try:
        import chronos  # pyright: ignore[reportMissingImports]
        import torch  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ModeloFundacionalNoDisponibleError(
            "el modelo fundacional requiere el extra 'foundation' "
            "(uv sync --extra foundation)"
        ) from exc
    return _cargar(config, chronos, torch)  # pragma: no cover


def _cargar(
    config: ConfiguracionFundacional, chronos: Any, torch: Any
) -> PipelineFundacional:  # pragma: no cover - requiere el modelo real (prueba slow)
    if chronos.__version__ != config.version_libreria:
        raise ModeloFundacionalNoDisponibleError(
            f"chronos-forecasting {chronos.__version__} instalado, pero la "
            f"configuracion fija {config.version_libreria}"
        )

    torch.set_num_threads(config.n_hilos)
    torch.use_deterministic_algorithms(True)

    try:
        pipeline = chronos.Chronos2Pipeline.from_pretrained(
            config.model_id, revision=config.revision, device_map=config.device
        )
    except Exception as exc:
        raise ModeloFundacionalNoDisponibleError(
            f"no se pudo cargar {config.model_id}@{config.revision}: {exc}"
        ) from exc

    modelo = pipeline.model
    modelo.to(dtype=getattr(torch, config.dtype))
    modelo.eval()
    for parametro in modelo.parameters():
        parametro.requires_grad_(False)

    _logger.info(
        "Modelo fundacional cargado %s@%s device=%s dtype=%s",
        config.model_id,
        config.revision,
        config.device,
        config.dtype,
    )
    return _AdaptadorChronos2(pipeline, config)
