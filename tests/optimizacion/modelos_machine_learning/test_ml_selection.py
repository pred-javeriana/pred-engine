"""seleccionar_configuracion_ml: HPO + Walk-Forward + LightGBM, extremo a extremo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pred_engine.optimizacion.control_reanudacion import (
    AlmacenManifiestosFs,
    EstadoCorrida,
    IncompatibilidadCorridaError,
)
from pred_engine.optimizacion.optimizadores.HPO.contratos import InfoTrial
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda
from pred_engine.optimizacion.optimizadores.modelos_machine_learning import (
    EspacioML,
    seleccionar_configuracion_ml,
    seleccionar_por_panel_ml,
)

# Espacio chico: cada trial ajusta LightGBM por ventana, asi que se acota.
_CHICO = EspacioML(
    lags_min=3,
    lags_max=5,
    n_estimators_min=5,
    n_estimators_max=15,
    max_depth_min=2,
    max_depth_max=3,
)
_KW = dict(espacio=_CHICO, n_trials=6, horizonte=7, paso=7, seed=0)


def _serie(n: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = 10 + 5 * np.sin(np.arange(n) * 2 * np.pi / 7)
    return np.maximum(0.0, base + rng.normal(0, 1, n))


def _panel(skus=("A", "B"), n: int = 60) -> pd.DataFrame:
    fechas = list(pd.date_range("2024-01-01", periods=n, freq="D"))
    return pd.DataFrame(
        {
            "sku_id": [s for s in skus for _ in range(n)],
            "timestamp": fechas * len(skus),
            "demand_qty": np.concatenate(
                [_serie(n, seed=i + 1) for i in range(len(skus))]
            ),
            "lead_time_days": [3] * (n * len(skus)),
        }
    )


def test_devuelve_la_configuracion_ganadora_con_todos_los_hiperparametros():
    resultado = seleccionar_configuracion_ml(_serie(), sku_id="S1", **_KW)
    sel = resultado.seleccionada
    assert sel is not None
    assert sel.sku_id == "S1"
    assert sel.metrica_objetivo == "mase"
    assert sel.valor > 0
    assert {"lags", "n_estimators", "max_depth", "learning_rate"} <= set(
        sel.hiperparametros
    )
    assert 3 <= sel.hiperparametros["lags"] <= 5
    assert resultado.estudio.trials[0].familia == "ml"


def test_mismo_seed_da_la_misma_configuracion_ganadora():
    a = seleccionar_configuracion_ml(_serie(), **_KW)
    b = seleccionar_configuracion_ml(_serie(), **_KW)
    assert a.seleccionada is not None and b.seleccionada is not None
    assert dict(a.seleccionada.hiperparametros) == dict(b.seleccionada.hiperparametros)
    assert a.seleccionada.valor == b.seleccionada.valor


def test_serie_insuficiente_falla_con_mensaje_accionable():
    with pytest.raises(ValueError, match="insuficiente para SKU='S9'"):
        seleccionar_configuracion_ml(_serie(n=10), sku_id="S9", **_KW)


def test_serie_no_1d_falla():
    with pytest.raises(ValueError, match="1D"):
        seleccionar_configuracion_ml(np.ones((30, 2)), **_KW)


def test_la_poda_queda_auditada_con_motivo():
    resultado = seleccionar_configuracion_ml(
        _serie(n=80),
        espacio=_CHICO,
        n_trials=20,
        horizonte=7,
        paso=7,
        reglas=ReglasPoda(min_ventanas=4, factor_reduccion=2),
    )
    podados = [t for t in resultado.estudio.trials if t.estado == "podado"]
    assert podados and all(t.motivo for t in podados)
    ids_descartadas = {t.id for t in resultado.descartadas}
    assert {t.id for t in podados} <= ids_descartadas


def test_warm_start_inyecta_trials_previos_sin_reevaluarlos():
    previo = InfoTrial(
        numero=0,
        estado="completado",
        valor=0.123,
        parametros={
            "lags": 4,
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.1,
            "min_child_weight": 2.0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 1e-3,
            "reg_lambda": 1e-3,
        },
        atributos={},
    )
    resultado = seleccionar_configuracion_ml(_serie(), historico_previo=[previo], **_KW)
    assert len(resultado.estudio.trials) == _KW["n_trials"] + 1
    assert 0.123 in {t.valor for t in resultado.estudio.trials}


def test_panel_ordena_skus_y_rechaza_run_id():
    resultado = seleccionar_por_panel_ml(_panel(("Z", "A")), **_KW)
    assert list(resultado) == ["A", "Z"]
    assert all(r.seleccionada is not None for r in resultado.values())
    with pytest.raises(ValueError, match="run_id"):
        seleccionar_por_panel_ml(_panel(), run_id="x", **_KW)


def test_n_procesos_invalido_falla_antes_de_calcular():
    with pytest.raises(ValueError, match="n_procesos"):
        seleccionar_por_panel_ml(_panel(), n_procesos=0, **_KW)


@pytest.mark.slow
def test_paralelo_entre_skus_da_el_mismo_resultado_que_secuencial():
    panel = _panel(("A", "B"))
    secuencial = seleccionar_por_panel_ml(panel, **_KW)
    paralelo = seleccionar_por_panel_ml(panel, n_procesos=2, **_KW)
    assert list(paralelo) == list(secuencial) == ["A", "B"]
    for sku in secuencial:
        s, p = secuencial[sku].seleccionada, paralelo[sku].seleccionada
        assert s is not None and p is not None
        assert dict(s.hiperparametros) == dict(p.hiperparametros)
        assert s.valor == p.valor


@pytest.mark.slow
def test_raiz_corrida_persiste_y_la_segunda_llamada_no_reejecuta(tmp_path):
    kw = dict(_KW, sku_id="S1", raiz_corrida=tmp_path, run_id="ml-S1-test")
    primera = seleccionar_configuracion_ml(_serie(), **kw)
    manifiesto = AlmacenManifiestosFs(tmp_path).cargar("ml-S1-test")
    assert manifiesto.estado is EstadoCorrida.COMPLETADA

    segunda = seleccionar_configuracion_ml(_serie(), **kw)
    assert primera.seleccionada is not None and segunda.seleccionada is not None
    assert dict(segunda.seleccionada.hiperparametros) == dict(
        primera.seleccionada.hiperparametros
    )
    assert [t.id for t in segunda.estudio.trials] == [
        t.id for t in primera.estudio.trials
    ]


@pytest.mark.slow
def test_reanudar_con_otra_serie_bajo_el_mismo_run_id_se_rechaza(tmp_path):
    kw = dict(_KW, sku_id="S1", raiz_corrida=tmp_path, run_id="ml-S1-test")
    seleccionar_configuracion_ml(_serie(seed=0), **kw)
    manifiesto = tmp_path / "ml-S1-test" / "manifiesto.json"
    antes = manifiesto.read_text()
    with pytest.raises(IncompatibilidadCorridaError):
        seleccionar_configuracion_ml(_serie(seed=1), **kw)
    assert manifiesto.read_text() == antes
