"""MuestreadorAleatorio (baseline) y MuestreadorTPE (informado)."""

from __future__ import annotations

import numpy as np

from pred_engine.comun.dataclasses.hpo import Trial
from pred_engine.optimizacion.optimizadores.HPO.espacio import (
    Categorico,
    Condicion,
    EspacioBusqueda,
    Flotante,
)
from pred_engine.optimizacion.optimizadores.HPO.muestreadores import (
    MuestreadorAleatorio,
    MuestreadorTPE,
)


def _espacio() -> EspacioBusqueda:
    return EspacioBusqueda(parametros=(Flotante("x", 0.0, 100.0),))


def _trial(x: float, valor: float) -> Trial:
    return Trial(
        id=f"t-{x}-{valor}",
        configuracion={"x": x},
        estado="completado",
        valor=valor,
        n_ventanas=10,
        motivo=None,
        metrica_objetivo="mase",
        familia="prueba",
    )


def test_aleatorio_reproducible_con_misma_semilla():
    m1 = MuestreadorAleatorio(_espacio(), seed=5)
    m2 = MuestreadorAleatorio(_espacio(), seed=5)
    sugeridos_1 = [m1.sugerir(()) for _ in range(10)]
    sugeridos_2 = [m2.sugerir(()) for _ in range(10)]
    assert sugeridos_1 == sugeridos_2


def test_aleatorio_cubre_el_rango():
    m = MuestreadorAleatorio(_espacio(), seed=1)
    valores = [m.sugerir(())["x"] for _ in range(200)]
    assert min(valores) < 20.0
    assert max(valores) > 80.0


def test_tpe_usa_respaldo_antes_del_arranque():
    tpe = MuestreadorTPE(_espacio(), n_arranque=5, seed=0)
    historial = [_trial(float(i), float(i)) for i in range(3)]
    sugerido = tpe.sugerir(historial)
    assert 0.0 <= sugerido["x"] <= 100.0


def test_tpe_concentra_sugerencias_cerca_del_optimo_conocido():
    espacio = _espacio()
    tpe = MuestreadorTPE(espacio, n_arranque=8, seed=1)
    rng = np.random.default_rng(1)
    historial: list[Trial] = []
    for _ in range(40):
        x = float(rng.uniform(0, 100))
        historial.append(_trial(x, abs(x - 70.0)))

    sugerencias = [tpe.sugerir(historial)["x"] for _ in range(30)]
    error_tpe = np.mean(np.abs(np.array(sugerencias) - 70.0))

    aleatorio = MuestreadorAleatorio(espacio, seed=1)
    referencia = [aleatorio.sugerir(historial)["x"] for _ in range(30)]
    error_aleatorio = np.mean(np.abs(np.array(referencia) - 70.0))

    assert error_tpe < error_aleatorio


def test_tpe_no_propone_parametros_inactivos():
    espacio = EspacioBusqueda(
        parametros=(Categorico("modo", ("a", "b")), Flotante("solo_b", 0.0, 1.0)),
        condiciones=(Condicion(parametro="solo_b", padre="modo", valores=("b",)),),
    )
    tpe = MuestreadorTPE(espacio, n_arranque=1, seed=0)
    historial = [
        Trial(
            id=f"t{i}",
            configuracion=(
                {"modo": "a"} if i % 2 == 0 else {"modo": "b", "solo_b": 0.5}
            ),
            estado="completado",
            valor=float(i),
            n_ventanas=5,
            motivo=None,
            metrica_objetivo="mase",
            familia="prueba",
        )
        for i in range(6)
    ]
    for _ in range(20):
        sugerido = tpe.sugerir(historial)
        if sugerido["modo"] == "a":
            assert "solo_b" not in sugerido
        else:
            assert "solo_b" in sugerido
