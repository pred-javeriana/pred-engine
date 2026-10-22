"""Limite de CPU de la sesion de pruebas, pensado para equipos de 4 nucleos.

Las bibliotecas de BLAS/OpenMP (NumPy, SciPy, statsmodels, LightGBM) leen sus
variables de hilos UNA VEZ, al importarse, y sin ellas abren un hilo por nucleo
en cada proceso. Este conftest raiz se carga antes de que una prueba importe
NumPy, asi que fija una hebra igual que el motor lo hace en cada hijo del pool.

Las pruebas que corren la CLI pasan ``--workers`` explicito (1 o 2): el valor
por defecto, ``nucleos - 1``, ocuparia la maquina entera con una sola prueba.
"""

from __future__ import annotations

from pred_engine.comun.ejecucion_paralela import fijar_una_hebra_blas

fijar_una_hebra_blas()
