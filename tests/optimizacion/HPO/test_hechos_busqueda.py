"""Una busqueda sin ensayos completos entrega sus hechos en campos (contrato 11.1).

La interfaz explica el fallo con estos campos, sin leer el mensaje: cuantos ensayos
fallaron y se podaron, y por cada causa su conteo y un ejemplo.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from tests._dobles import fabrica_nivel_constante, fabrica_rota

from pred_engine.comun.hechos_error import hechos_de
from pred_engine.comun.modelos.modelos_clasicos import sarima
from pred_engine.comun.modelos.modelos_clasicos.sarima import SarimaForecaster
from pred_engine.optimizacion.optimizadores.HPO.asha import DecisorASHA
from pred_engine.optimizacion.optimizadores.HPO.errores import EstudioError
from pred_engine.optimizacion.optimizadores.HPO.espacio import (
    EspacioBusqueda,
    Flotante,
    nombre_restriccion,
)
from pred_engine.optimizacion.optimizadores.HPO.estudio import ejecutar_estudio
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda
from pred_engine.optimizacion.optimizadores.modelos_deep_learning.espacio_dl import (
    construir_espacio_dl,
)


def limite_nivel(configuracion):
    return configuracion["nivel"] < 0


def _serie() -> np.ndarray:
    return 10 + np.sin(np.arange(40, dtype=float))


def test_restriccion_que_rechaza_todo_da_su_nombre_corto_y_los_intentos():
    with pytest.raises(EstudioError) as error:
        ejecutar_estudio(
            _serie(),
            EspacioBusqueda(
                parametros=(Flotante("nivel", 1, 2),),
                restricciones=(limite_nivel,),
            ),
            fabrica_rota,
            n_trials=1,
            min_train=20,
        )

    assert hechos_de(error.value) == {
        "search": {
            "failed": 21,
            "pruned": 0,
            "invalid_attempts": 21,
            "causes": [
                {
                    "cause": "configuracion_invalida",
                    "count": 21,
                    "example": {"constraint": "limite_nivel"},
                }
            ],
        }
    }


def test_ajuste_fallido_da_la_ventana_y_por_que_no_ajusto_el_modelo(monkeypatch):
    # Con un modulo minimo inalcanzable, todo ajuste SARIMA real queda "en el borde".
    monkeypatch.setattr(sarima, "MODULO_MINIMO_RAIZ", 1e6)

    def fabrica(configuracion: Mapping[str, Any], *, seed: int) -> SarimaForecaster:
        return SarimaForecaster((1, 0, 0), (0, 0, 0, 0), seed=seed)

    estudio = ejecutar_estudio(
        _serie(),
        EspacioBusqueda(parametros=(Flotante("nivel", 1, 2),)),
        fabrica,
        n_trials=2,
        min_train=20,
    )

    busqueda = estudio.hechos_busqueda()["search"]
    assert (busqueda["failed"], busqueda["pruned"]) == (2, 0)
    [causa] = busqueda["causes"]
    assert (causa["cause"], causa["count"]) == ("ajuste_fallido", 2)
    ejemplo = causa["example"]
    assert ejemplo["window"] == 1
    ajuste = ejemplo["fit"]
    assert ajuste["model"] == "SARIMA(1, 0, 0)x(0, 0, 0, 0)"
    assert ajuste["reason"] == "raiz_en_borde"
    assert ajuste["limit"] == 1e6
    assert 0 < ajuste["value"] < 1e6


def test_un_ajuste_sin_hechos_conserva_la_ventana():
    estudio = ejecutar_estudio(
        _serie(),
        EspacioBusqueda(parametros=(Flotante("nivel", 1, 2),)),
        fabrica_rota,
        n_trials=2,
        min_train=20,
    )

    assert estudio.hechos_busqueda()["search"]["causes"] == [
        {"cause": "ajuste_fallido", "count": 2, "example": {"window": 1}}
    ]


def test_poda_semantica_da_la_regla_y_el_error_de_la_ventana():
    # Un pronostico de cero sobre una serie positiva es degenerado (regla #3).
    estudio = ejecutar_estudio(
        _serie(),
        EspacioBusqueda(parametros=(Flotante("nivel", 0, 1e-12),)),
        fabrica_nivel_constante,
        n_trials=2,
        min_train=20,
    )

    [causa] = estudio.hechos_busqueda()["search"]["causes"]
    assert (causa["cause"], causa["count"]) == ("poda_semantica", 2)
    ejemplo = causa["example"]
    assert ejemplo["window"] == 1
    assert ejemplo["rule"] == "prediccion_nula"
    assert ejemplo["value"] > 0


def test_la_poda_por_escalon_da_el_escalon_el_error_y_el_umbral():
    decisor = DecisorASHA(
        ReglasPoda(min_ventanas=4, factor_reduccion=3), n_ventanas_totales=30
    )

    decision = decisor.decidir(
        n_evaluadas=4,
        numero_trial=2,
        valores_competidores_en_escalon={0: 1.0, 1: 2.0, 2: 3.0},
    )

    assert decision.podar
    assert decision.hechos == {"rung": 4, "value": 3.0, "threshold": 1.0, "keep": 1}


def test_las_causas_van_de_la_mas_frecuente_a_la_menos():
    estudio = ejecutar_estudio(
        _serie(),
        EspacioBusqueda(parametros=(Flotante("nivel", 1, 2),)),
        fabrica_rota,
        n_trials=3,
        min_train=20,
    )
    estudio = replace(
        estudio,
        causas={"escalon": 1, "ajuste_fallido": 3, "configuracion_invalida": 1},
    )

    assert [c["cause"] for c in estudio.hechos_busqueda()["search"]["causes"]] == [
        "ajuste_fallido",
        "configuracion_invalida",
        "escalon",
    ]


def test_nombre_corto_de_una_restriccion():
    assert nombre_restriccion(limite_nivel) == "limite_nivel"
    [costo] = construir_espacio_dl().restricciones
    assert nombre_restriccion(costo) == "costo_acotado"
    assert nombre_restriccion(lambda c: True) is None
