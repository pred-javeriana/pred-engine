"""`ejecutar_estudio`: el punto de entrada unico del HPO, extremo a extremo."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest
from tests._dobles import (
    fabrica_falla_primeras_n,
    fabrica_nivel_constante,
    fabrica_rota,
)

from pred_engine.optimizacion.optimizadores.HPO.contratos import InfoTrial
from pred_engine.optimizacion.optimizadores.HPO.errores import EstudioError
from pred_engine.optimizacion.optimizadores.HPO.espacio import (
    Categorico,
    EspacioBusqueda,
    Flotante,
)
from pred_engine.optimizacion.optimizadores.HPO.estudio import ejecutar_estudio
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda


def _serie_objetivo(nivel: float = 50.0, n: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return nivel + rng.normal(0, 1.0, size=n)


def _espacio() -> EspacioBusqueda:
    return EspacioBusqueda(parametros=(Flotante("nivel", 0.0, 100.0),))


def test_estudio_encuentra_el_optimo_conocido():
    y = _serie_objetivo(nivel=50.0)
    resultado = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_nivel_constante,
        n_trials=60,
        min_train=20,
        horizonte=1,
        paso=1,
        estacionalidad=1,
        seed=0,
        familia="prueba",
    )
    assert resultado.mejor is not None
    assert abs(resultado.mejor.configuracion["nivel"] - 50.0) < 5.0


def test_todos_los_trials_comparten_las_mismas_ventanas():
    y = _serie_objetivo()
    resultado = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_nivel_constante,
        n_trials=10,
        min_train=20,
        estacionalidad=1,
        seed=1,
    )
    assert len(resultado.ventanas) == len(y) - 20
    assert all(t.n_ventanas <= len(resultado.ventanas) for t in resultado.trials)


def test_determinismo_extremo_a_extremo():
    y = _serie_objetivo()
    r1 = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_nivel_constante,
        n_trials=15,
        min_train=20,
        estacionalidad=1,
        seed=42,
    )
    r2 = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_nivel_constante,
        n_trials=15,
        min_train=20,
        estacionalidad=1,
        seed=42,
    )
    assert r1.mejor is not None and r2.mejor is not None
    assert r1.mejor.configuracion == r2.mejor.configuracion
    assert r1.mejor.valor == r2.mejor.valor


def test_mejor_es_none_cuando_todo_falla():
    y = _serie_objetivo()
    resultado = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_rota,
        n_trials=5,
        min_train=20,
        estacionalidad=1,
        seed=0,
    )
    assert resultado.mejor is None
    assert resultado.n_completados == 0
    assert resultado.n_fallidos == 5


def test_fallo_en_las_primeras_ventanas_no_mata_el_trial_si_luego_converge():
    y = _serie_objetivo(n=60)
    resultado = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_falla_primeras_n(3),
        n_trials=1,
        min_train=20,
        horizonte=1,
        paso=1,
        estacionalidad=1,
        seed=0,
        familia="prueba",
    )
    trial = resultado.trials[0]
    assert trial.estado == "completado"
    assert trial.valor is not None
    assert math.isfinite(trial.valor)
    assert resultado.n_fallidos == 0


def test_fallo_en_todas_las_ventanas_hasta_la_poda_se_marca_fallido_no_podado():
    y = _serie_objetivo()
    resultado = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_rota,
        n_trials=3,
        min_train=20,
        estacionalidad=1,
        seed=0,
    )
    assert resultado.n_podados == 0
    assert resultado.n_fallidos == 3
    assert all(t.estado == "fallido" for t in resultado.trials)


def test_configuracion_invalida_no_quema_presupuesto_de_trials_reales():
    """Antes, cada configuracion rechazada por una restriccion consumia un
    `indice_trial` completo marcado FAIL sin evaluar nada -- con un espacio
    donde el 90% de las combinaciones es invalido, varios de los `n_trials`
    pedidos jamas llegaban a correr el walk-forward. Ahora esos intentos
    invalidos no cuentan contra el presupuesto: se pide `n_trials=5` y se
    obtienen 5 trials REALMENTE evaluados, mas los intentos invalidos
    descartados aparte (identificables por su motivo)."""
    y = _serie_objetivo()
    espacio = EspacioBusqueda(
        parametros=(
            Flotante("nivel", 0.0, 100.0),
            Categorico("modo", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),
        ),
        # solo 1 de cada 10 valores de "modo" es valido: ~90% de rechazo
        restricciones=(lambda c: c["modo"] == 1,),
    )
    resultado = ejecutar_estudio(
        y,
        espacio,
        fabrica_nivel_constante,
        n_trials=5,
        min_train=20,
        estacionalidad=1,
        seed=3,
    )
    reales = [t for t in resultado.trials if t.motivo != "configuracion_invalida"]
    invalidos = [t for t in resultado.trials if t.motivo == "configuracion_invalida"]
    assert len(reales) == 5
    assert len(invalidos) > 0


def test_espacio_casi_totalmente_invalido_lanza_estudioerror():
    """Techo de seguridad: si las restricciones rechazan casi todo el
    espacio, el estudio debe fallar explicitamente en vez de intentar para
    siempre (o de gastar miles de intentos en silencio)."""
    y = _serie_objetivo()
    espacio = EspacioBusqueda(
        parametros=(Flotante("nivel", 0.0, 100.0),),
        restricciones=(lambda c: False,),  # rechaza absolutamente todo
    )
    with pytest.raises(EstudioError):
        ejecutar_estudio(
            y,
            espacio,
            fabrica_nivel_constante,
            n_trials=2,
            min_train=20,
            estacionalidad=1,
            seed=0,
        )


def test_cada_trial_registra_timestamp_iso8601():
    y = _serie_objetivo()
    resultado = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_nivel_constante,
        n_trials=3,
        min_train=20,
        estacionalidad=1,
        seed=0,
    )
    assert len(resultado.trials) == 3
    for trial in resultado.trials:
        assert trial.timestamp is not None
        datetime.fromisoformat(trial.timestamp)  # no lanza si es ISO-8601 valido


def test_warm_start_inyecta_historico_sin_reevaluarlo():
    # El historico "previo" simula trials ya evaluados en OTRA corrida
    # (ej. un segmento anterior de la misma serie, SKU lumpy) -- deben
    # aparecer en el resultado sin pasar por `fabrica_nivel_constante` de
    # nuevo: si `fabrica_rota` se usara aca en su lugar, un trial
    # reevaluado fallaria y esta prueba lo detectaria.
    y = _serie_objetivo()
    historico = [
        InfoTrial(
            numero=0,
            estado="completado",
            valor=1.23,
            parametros={"nivel": 42.0},
            atributos={},
        ),
        InfoTrial(
            numero=1,
            estado="completado",
            valor=4.56,
            parametros={"nivel": 7.0},
            atributos={},
        ),
    ]
    resultado = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_rota,  # si se reevaluaran, marcarian n_fallidos > 0
        n_trials=3,
        min_train=20,
        estacionalidad=1,
        seed=0,
        historico_previo=historico,
    )
    assert len(resultado.trials) == 3 + 2
    valores = {t.valor for t in resultado.trials}
    assert {1.23, 4.56} <= valores
    assert resultado.n_fallidos == 3  # solo los 3 trials NUEVOS (fabrica_rota)


def test_warm_start_con_espacio_incompatible_no_falla_el_estudio():
    y = _serie_objetivo()
    historico = [
        InfoTrial(
            numero=0,
            estado="completado",
            valor=1.0,
            parametros={"parametro_que_no_existe": 1},
            atributos={},
        )
    ]
    resultado = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_nivel_constante,
        n_trials=2,
        min_train=20,
        estacionalidad=1,
        seed=0,
        historico_previo=historico,
    )
    assert len(resultado.trials) == 2  # el trial incompatible se descarto


def test_poda_reduce_las_ventanas_evaluadas_respecto_al_modo_sin_poda():
    y = _serie_objetivo()
    reglas_con_poda = ReglasPoda(min_ventanas=4, factor_reduccion=2)
    resultado = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_nivel_constante,
        n_trials=30,
        min_train=20,
        estacionalidad=1,
        seed=7,
        reglas=reglas_con_poda,
    )
    assert resultado.n_podados > 0
    ventanas_evaluadas = sum(t.n_ventanas for t in resultado.trials)
    ventanas_sin_poda = len(resultado.ventanas) * len(resultado.trials)
    assert ventanas_evaluadas < ventanas_sin_poda


def test_asha_reduce_el_costo_de_evaluacion_en_al_menos_30_por_ciento():
    """TASK-HPO-5.0-C1. Se usa 'ventanas evaluadas' (no tiempo de reloj)
    como proxy del costo computacional: cada ventana walk-forward evaluada
    es un ajuste+prediccion real, proporcional al tiempo de computo, sin la
    varianza de medir reloj en un CI compartido."""
    y = _serie_objetivo(n=60)
    reglas_con_poda = ReglasPoda(min_ventanas=4, factor_reduccion=2)
    resultado = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_nivel_constante,
        n_trials=30,
        min_train=20,
        estacionalidad=1,
        seed=7,
        reglas=reglas_con_poda,
    )
    ventanas_evaluadas = sum(t.n_ventanas for t in resultado.trials)
    ventanas_sin_poda = len(resultado.ventanas) * len(resultado.trials)
    reduccion = 1 - (ventanas_evaluadas / ventanas_sin_poda)
    assert reduccion >= 0.30


def test_todos_los_trials_podados_tienen_motivo_auditable():
    """TASK-HPO-5.0-C1: el registro de auditoria debe justificar cada poda."""
    y = _serie_objetivo(n=60)
    reglas_con_poda = ReglasPoda(min_ventanas=4, factor_reduccion=2)
    resultado = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_nivel_constante,
        n_trials=30,
        min_train=20,
        estacionalidad=1,
        seed=7,
        reglas=reglas_con_poda,
    )
    podados = [t for t in resultado.trials if t.estado == "podado"]
    assert len(podados) > 0
    assert all(t.motivo for t in podados)
