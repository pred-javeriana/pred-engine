"""Configuracion congelada del modelo fundacional .

La familia fundacional NO se optimiza: no hay espacio de busqueda, HPO ni
fine-tuning. Esta es su unica configuracion, identica para todo SKU y todo
perfil topologico.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ConfiguracionFundacional:
    model_id: str
    revision: str
    version_libreria: str
    device: Literal["cpu"] = "cpu"
    dtype: Literal["float32"] = "float32"
    cuantil_puntual: float = 0.5
    aprendizaje_cruzado: bool = False
    max_contexto: int = 8192
    n_hilos: int = 1

    def descripcion_canonica(self) -> dict[str, Any]:
        """Representacion JSON-serializable y estable (payload y trazabilidad)."""
        return asdict(self)


CHRONOS2_ZERO_SHOT = ConfiguracionFundacional(
    model_id="amazon/chronos-2",
    revision="29ec3766d36d6f73f0696f85560a422f50e8498c",
    version_libreria="2.3.2",
)
