"""Seleccion DL: declara espacio/fabrica y delega TPE + ASHA + Walk-Forward."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pred_engine.comun.dataclasses.hpo import ResultadoEstudio
from pred_engine.comun.modelos.modelos_deep_learning import fabrica_dl
from pred_engine.comun.walkforward.ventanas import generar_ventanas
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda
from pred_engine.optimizacion.optimizadores.modelos_deep_learning.espacio_dl import (
    EspacioDL,
    construir_espacio_dl,
    min_train_recomendado_dl,
)
from pred_engine.optimizacion.optimizadores.seleccion_hpo import (
    exigir_serie,
    seleccionar_con_hpo,
)

FAMILIA_DL = "dl"

# La poda semantica del motor comun puede cerrar en la primera ventana. DL
# usa solo ASHA para respetar el minimo de cuatro ventanas sin cambiar el motor.
REGLAS_DL = ReglasPoda(habilitar_poda_semantica=False)


def seleccionar_configuracion_dl(
    y: np.ndarray,
    *,
    sku_id: str | None = None,
    espacio: EspacioDL | None = None,
    n_trials: int = 12,
    min_train: int | None = None,
    horizonte: int = 7,
    paso: int = 7,
    metrica_objetivo: str = "rmse",
    estacionalidad: int = 1,
    muestreador: Any | None = None,
    reglas: ReglasPoda | None = None,
    seed: int = 0,
    raiz_corrida: str | Path | None = None,
    run_id: str | None = None,
) -> ResultadoEstudio:
    """Devuelve el resultado del estudio comun, sin un protocolo paralelo DL.

    Se exige historia para entrenar cualquier candidato y completar al menos
    `reglas.min_ventanas` ventanas causales. No se reduce el espacio en silencio.
    """
    cfg = espacio or EspacioDL()
    reglas = reglas if reglas is not None else REGLAS_DL
    minimo_modelo = min_train_recomendado_dl(cfg)
    minimo = (
        min_train if min_train is not None else max(minimo_modelo, estacionalidad + 1)
    )
    for nombre, valor in (
        ("n_trials", n_trials),
        ("min_train", minimo),
        ("horizonte", horizonte),
        ("paso", paso),
        ("estacionalidad", estacionalidad),
    ):
        if type(valor) is not int or valor < 1:
            raise ValueError(f"{nombre} debe ser un entero positivo")
    if minimo < minimo_modelo:
        raise ValueError(
            f"min_train insuficiente: se requieren al menos {minimo_modelo}"
        )
    if metrica_objetivo == "mase" and minimo <= estacionalidad:
        raise ValueError("min_train debe superar estacionalidad para MASE")
    if type(reglas.min_ventanas) is not int or reglas.min_ventanas < 4:
        raise ValueError("DL requiere min_ventanas >= 4")
    if type(reglas.factor_reduccion) is not int or reglas.factor_reduccion < 2:
        raise ValueError("factor_reduccion debe ser un entero >= 2")
    if reglas.habilitar_poda_semantica:
        raise ValueError(
            "DL requiere habilitar_poda_semantica=False: poda solo con ASHA"
        )
    serie = exigir_serie(y, min_train=minimo, horizonte=horizonte, sku_id=sku_id)
    if not np.isfinite(serie).all():
        raise ValueError("y contiene valores no finitos")
    ventanas = generar_ventanas(
        len(serie), min_train=minimo, horizonte=horizonte, paso=paso
    )
    if len(ventanas) < reglas.min_ventanas:
        raise ValueError(
            f"historial insuficiente para SKU={sku_id!r}: {len(ventanas)} ventanas; "
            f"se requieren al menos {reglas.min_ventanas}"
        )
    return seleccionar_con_hpo(
        serie,
        construir_espacio_dl(cfg),
        fabrica_dl,
        familia=FAMILIA_DL,
        sku_id=sku_id,
        n_trials=n_trials,
        min_train=minimo,
        horizonte=horizonte,
        paso=paso,
        metrica_objetivo=metrica_objetivo,
        estacionalidad=estacionalidad,
        muestreador=muestreador,
        reglas=reglas,
        seed=seed,
        raiz_corrida=raiz_corrida,
        run_id=run_id,
    )
