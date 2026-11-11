"""Los errores que conocen sus hechos los entregan en campos (contrato 11.1)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from pred_engine.comun.hechos_error import ajuste, hechos_de
from pred_engine.comun.modelos.modelos_clasicos.errores import AjusteModeloError
from pred_engine.comun.modelos.modelos_machine_learning.errores import SerieCortaError
from pred_engine.comun.modelos.modelos_machine_learning.lgbm import LightGBMForecaster
from pred_engine.comun.walkforward.errores import VentanaInsuficienteError
from pred_engine.comun.walkforward.ventanas import generar_ventanas
from pred_engine.optimizacion.router.errores import SelectionContractError


def test_una_serie_corta_para_la_validacion_da_lo_exigido_y_lo_disponible():
    with pytest.raises(VentanaInsuficienteError) as error:
        generar_ventanas(10, min_train=12, horizonte=2)

    assert hechos_de(error.value) == {"history": {"required": 14, "available": 10}}


def test_una_serie_corta_para_lightgbm_da_lo_exigido_y_lo_disponible():
    with pytest.raises(SerieCortaError) as error:
        LightGBMForecaster(lags=7).fit(np.ones(5))

    hechos = hechos_de(error.value)
    assert hechos is not None
    assert hechos["history"]["available"] == 5
    assert hechos["history"]["required"] > 5


def test_un_error_que_envuelve_a_otro_conserva_los_hechos_de_la_causa():
    try:
        try:
            generar_ventanas(10, min_train=12, horizonte=2)
        except ValueError as exc:
            raise SelectionContractError("seleccion imposible") from exc
    except SelectionContractError as envoltorio:
        assert hechos_de(envoltorio) == {"history": {"required": 14, "available": 10}}


def test_un_error_sin_hechos_no_inventa_ninguno():
    assert hechos_de(RuntimeError("sin datos")) is None


def test_los_hechos_omiten_vacios_y_numeros_no_finitos():
    assert ajuste("raiz_en_borde", modelo=None, valor=float("nan"), limite=1.01) == {
        "fit": {"reason": "raiz_en_borde", "limit": 1.01}
    }


def test_los_hechos_sobreviven_al_paso_entre_procesos():
    error = AjusteModeloError("no ajusto", hechos=ajuste("sin_convergencia"))

    copia = pickle.loads(pickle.dumps(error))

    assert str(copia) == "no ajusto"
    assert hechos_de(copia) == {"fit": {"reason": "sin_convergencia"}}
