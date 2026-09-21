"""Benchmarks del motor de HPO contra funciones objetivo sinteticas.

Distinto del resto de `optimizadores/HPO/`: estas funciones NO tienen
dependencia temporal ni pasan por Walk-Forward -- son el benchmark estandar
de la literatura de optimizacion global (sphere/Rastrigin/Rosenbrock), usado
aqui solo para comparar la velocidad de convergencia de los samplers
(TASK-HPO-4.0-C1), no el comportamiento de PRED sobre series de tiempo.
"""

from __future__ import annotations
