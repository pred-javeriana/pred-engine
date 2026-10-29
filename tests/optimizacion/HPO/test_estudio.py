"""`ejecutar_estudio`: el punto de entrada unico del HPO, extremo a extremo."""

from __future__ import annotations

import json
import math
from datetime import datetime

import numpy as np
import optuna
import pytest
from tests._dobles import (
    fabrica_falla_primeras_n,
    fabrica_nivel_constante,
    fabrica_rota,
)

from pred_engine.comun.ensayos_hpo import leer_ensayos
from pred_engine.optimizacion.optimizadores.HPO.adaptador_optuna import (
    crear_estudio_optuna,
)
from pred_engine.optimizacion.optimizadores.HPO.asha import DecisorASHA
from pred_engine.optimizacion.optimizadores.HPO.contratos import InfoTrial
from pred_engine.optimizacion.optimizadores.HPO.errores import EstudioError
from pred_engine.optimizacion.optimizadores.HPO.espacio import (
    Categorico,
    EspacioBusqueda,
    Flotante,
)
from pred_engine.optimizacion.optimizadores.HPO.estudio import ejecutar_estudio
from pred_engine.optimizacion.optimizadores.HPO.muestreadores import (
    construir_muestreador_tpe,
)
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


@pytest.mark.parametrize(
    ("n_observaciones", "estado_segundo", "ventanas_segundo", "completados", "podados"),
    [(24, "completado", 4, 2, 0), (28, "podado", 4, 1, 1)],
)
def test_asha_solo_poda_si_quedan_ventanas_por_evaluar(
    n_observaciones, estado_segundo, ventanas_segundo, completados, podados
):
    # La semilla produce primero un nivel cercano a 50 y luego uno peor.
    # Con cuatro ventanas, ASHA solo podria descartar tras evaluar todas.
    resultado = ejecutar_estudio(
        np.full(n_observaciones, 50.0),
        _espacio(),
        fabrica_nivel_constante,
        n_trials=2,
        min_train=20,
        metrica_objetivo="mae",
        estacionalidad=1,
        muestreador=optuna.samplers.RandomSampler(seed=0),
        reglas=ReglasPoda(min_ventanas=4),
    )

    primero, segundo = resultado.trials
    assert primero.estado == "completado"
    assert primero.n_ventanas == n_observaciones - 20
    assert segundo.estado == estado_segundo
    assert segundo.n_ventanas == ventanas_segundo
    assert resultado.n_completados == completados
    assert resultado.n_podados == podados
    assert resultado.n_fallidos == 0
    assert resultado.mejor is not None
    assert resultado.mejor.id == primero.id


def test_etiqueta_del_motor_preserva_el_muestreo_tpe_y_ambos_lectores(tmp_path):
    reglas = ReglasPoda(min_ventanas=4)
    referencia, _ = crear_estudio_optuna(
        muestreador=construir_muestreador_tpe(seed=0),
        decisor_asha=DecisorASHA(reglas, n_ventanas_totales=4),
    )
    # Recorrido anterior: incluso en la cuarta ventana se conserva PRUNED
    # para el backend. MAE del pronostico constante sobre demanda 50.
    for _ in range(30):
        trial = referencia.ask()
        nivel = trial.suggest_float("nivel", 0.0, 100.0)
        valor = abs(50.0 - nivel)
        for ventana in range(1, 5):
            trial.report(valor, step=ventana)
            if trial.should_prune():
                referencia.tell(trial, None, estado="podado")
                break
        else:
            referencia.tell(trial, valor, estado="completado")

    resultado = ejecutar_estudio(
        np.full(24, 50.0),
        _espacio(),
        fabrica_nivel_constante,
        n_trials=30,
        min_train=20,
        metrica_objetivo="mae",
        estacionalidad=1,
        seed=0,
        reglas=reglas,
        raiz_corrida=tmp_path,
        run_id="tpe",
    )
    anteriores = referencia.trials_finalizados()
    assert [t.configuracion for t in resultado.trials] == [
        dict(t.parametros) for t in anteriores
    ]
    podados_backend = [t.numero for t in anteriores if t.estado == "podado"]
    assert len(podados_backend) > 0
    persistidos = leer_ensayos(tmp_path / "tpe")
    backend = [
        json.loads(linea)
        for linea in (tmp_path / "tpe" / "backend.jsonl").read_text().splitlines()
    ]
    for indice in podados_backend:
        assert backend[indice]["estado_optuna"] == "PRUNED"
        assert backend[indice]["user_attrs"]["estado_motor"] == "completado"
        for ensayo in (resultado.trials[indice], persistidos[indice]):
            assert ensayo.estado == "completado"
            assert ensayo.n_ventanas == 4
            assert ensayo.motivo is None
    assert resultado.n_completados == 30
    assert resultado.n_podados == 0


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


def test_un_ajuste_fallido_en_una_ventana_cierra_el_trial_como_fallido():
    # ADR-020: todos los trials se comparan en las mismas ventanas; uno que no
    # ajusta en alguna no es elegible aunque converja en las siguientes.
    y = _serie_objetivo(n=60)
    resultado = ejecutar_estudio(
        y,
        _espacio(),
        fabrica_falla_primeras_n(1),
        n_trials=2,
        min_train=20,
        horizonte=1,
        paso=1,
        estacionalidad=1,
        seed=0,
        familia="prueba",
    )
    primero, segundo = resultado.trials
    assert primero.estado == "fallido"
    assert primero.valor is None
    assert primero.n_ventanas == 1
    assert primero.motivo is not None and "ajuste_fallido" in primero.motivo
    assert segundo.estado == "completado"
    assert segundo.valor is not None and math.isfinite(segundo.valor)
    assert resultado.mejor is not None
    assert resultado.mejor.id == segundo.id


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
    reales = [
        t
        for t in resultado.trials
        if not (t.motivo or "").startswith("configuracion_invalida:")
    ]
    invalidos = [
        t
        for t in resultado.trials
        if (t.motivo or "").startswith("configuracion_invalida:")
    ]
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


class _NivelQueDesborda:
    """Nivel constante; sobre 60 pronostica 1e200, finito, pero su RMSE desborda."""

    def __init__(self, nivel: float) -> None:
        self.nivel = nivel

    def fit(self, y: np.ndarray) -> _NivelQueDesborda:
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, 1e200 if self.nivel > 60 else self.nivel)


def test_un_trial_que_asha_cierra_en_la_ultima_ventana_sin_valor_finito_queda_fallido():
    # Con 4 ventanas y min_ventanas=4, ASHA decide por primera vez en la ultima. Un
    # trial sin ninguna ventana finita llega ahi con un valor parcial no finito.
    resultado = ejecutar_estudio(
        np.full(24, 50.0),
        _espacio(),
        lambda configuracion, *, seed: _NivelQueDesborda(configuracion["nivel"]),
        n_trials=30,
        min_train=20,
        metrica_objetivo="rmse",
        estacionalidad=1,
        seed=0,
        reglas=ReglasPoda(min_ventanas=4, habilitar_poda_semantica=False),
    )
    desbordados = [t for t in resultado.trials if t.configuracion["nivel"] > 60]
    assert len(desbordados) > 0
    assert all(t.estado == "fallido" and t.motivo for t in desbordados)
    assert resultado.n_completados + resultado.n_podados + resultado.n_fallidos == 30
