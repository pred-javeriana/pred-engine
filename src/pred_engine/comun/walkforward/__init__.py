"""Definicion causal unica de Walk-Forward Validation (componentes 8.1 y 8.2).

Un solo `generar_ventanas()` alimenta tanto el modo integro
(`evaluar_walk_forward`) como el greedy (`iterar_walk_forward` /
`EjecutorGreedy`), para que sus metricas sean comparables entre si (ver
docs/adr/ADR-003). No depende de `pred_engine.forecasting` ni de ningun
motor de modelos concreto: habla con ellos via `Pronosticador` /
`FabricaPronosticador` (Protocol estructural).
"""

from pred_engine.comun.walkforward.errores import (
    EvaluacionError,
    VentanaInsuficienteError,
)
from pred_engine.comun.walkforward.metricas import (
    agregar,
    calcular_metricas,
    mae,
    mase,
    rmse,
    smape,
    valor_agregado_de,
)
from pred_engine.comun.walkforward.protocolos import (
    FabricaPronosticador,
    Pronosticador,
)
from pred_engine.comun.walkforward.semillas import derivar_semilla
from pred_engine.comun.walkforward.ventanas import generar_ventanas
from pred_engine.comun.walkforward.walk_forward import evaluar_walk_forward
from pred_engine.comun.walkforward.walk_forward_greedy import (
    EjecutorGreedy,
    iterar_walk_forward,
)

__all__ = [
    "EjecutorGreedy",
    "EvaluacionError",
    "FabricaPronosticador",
    "Pronosticador",
    "VentanaInsuficienteError",
    "agregar",
    "calcular_metricas",
    "derivar_semilla",
    "evaluar_walk_forward",
    "generar_ventanas",
    "iterar_walk_forward",
    "mae",
    "mase",
    "rmse",
    "smape",
    "valor_agregado_de",
]
