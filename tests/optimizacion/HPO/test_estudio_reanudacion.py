"""Integracion 2.9-C2: interrupcion simulada vs corrida continua."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests._dobles import fabrica_nivel_constante

from pred_engine.optimizacion.control_reanudacion import (
    AlmacenManifiestosFs,
    EstadoCorrida,
)
from pred_engine.optimizacion.optimizadores.HPO.espacio import EspacioBusqueda, Flotante
from pred_engine.optimizacion.optimizadores.HPO.estudio import ejecutar_estudio
from pred_engine.optimizacion.optimizadores.HPO.muestreadores import (
    construir_muestreador_aleatorio,
)


def _serie() -> np.ndarray:
    rng = np.random.default_rng(0)
    return 50.0 + rng.normal(0, 1.0, size=40)


def _espacio() -> EspacioBusqueda:
    return EspacioBusqueda(parametros=(Flotante("nivel", 0.0, 100.0),))


def _kwargs(tmp_path: Path, run_id: str, n_trials: int = 6) -> dict[str, object]:
    return {
        "y": _serie(),
        "espacio": _espacio(),
        "fabrica": fabrica_nivel_constante,
        "n_trials": n_trials,
        "min_train": 20,
        "horizonte": 1,
        "paso": 1,
        "estacionalidad": 1,
        "seed": 0,
        "familia": "prueba",
        "sku_id": "sku-1",
        "muestreador": construir_muestreador_aleatorio(seed=0),
        "raiz_corrida": tmp_path,
        "run_id": run_id,
    }


class _FabricaBomba:
    """Envoltorio que explota al iniciar el trial N (1-indexado evaluados)."""

    def __init__(self, n_explotar: int) -> None:
        self.n_explotar = n_explotar
        self.evaluados = 0
        self._config_actual: tuple[tuple[str, object], ...] | None = None

    def __call__(self, configuracion, *, seed: int):
        inner = fabrica_nivel_constante(configuracion, seed=seed)
        padre = self
        config_key = tuple(sorted(configuracion.items()))

        class _Proxy:
            def fit(self, y):
                if padre._config_actual != config_key:
                    padre._config_actual = config_key
                    padre.evaluados += 1
                    if padre.evaluados == padre.n_explotar:
                        raise KeyboardInterrupt("interrupcion simulada")
                return inner.fit(y)

            def predict(self, horizon: int):
                return inner.predict(horizon)

        return _Proxy()


def test_sin_controlador_no_escribe(tmp_path: Path) -> None:
    ejecutar_estudio(
        _serie(),
        _espacio(),
        fabrica_nivel_constante,
        n_trials=2,
        min_train=20,
        estacionalidad=1,
        seed=0,
        muestreador=construir_muestreador_aleatorio(seed=0),
    )
    assert list(tmp_path.iterdir()) == []


def test_corrida_nueva_queda_completada(tmp_path: Path) -> None:
    resultado = ejecutar_estudio(**_kwargs(tmp_path, "r1", n_trials=4))
    manifiesto = AlmacenManifiestosFs(tmp_path).cargar("r1")
    assert manifiesto.estado is EstadoCorrida.COMPLETADA
    assert manifiesto.n_trials_finalizados == 4
    assert resultado.n_completados + resultado.n_podados + resultado.n_fallidos >= 4


def test_completada_no_repite_trials(tmp_path: Path) -> None:
    primero = ejecutar_estudio(**_kwargs(tmp_path, "r1", n_trials=4))
    segundo = ejecutar_estudio(**_kwargs(tmp_path, "r1", n_trials=4))
    assert [t.id for t in primero.trials] == [t.id for t in segundo.trials]
    assert [t.valor for t in primero.trials] == [t.valor for t in segundo.trials]


def test_interrupcion_y_recuperacion_misma_frontera(tmp_path: Path) -> None:
    continuo = ejecutar_estudio(**_kwargs(tmp_path, "continua", n_trials=6))
    bomba = _FabricaBomba(n_explotar=3)
    kwargs = _kwargs(tmp_path, "partida", n_trials=6)
    kwargs["fabrica"] = bomba
    with pytest.raises(KeyboardInterrupt):
        ejecutar_estudio(**kwargs)

    manifiesto = AlmacenManifiestosFs(tmp_path).cargar("partida")
    assert manifiesto.estado is EstadoCorrida.INTERRUMPIDA
    assert manifiesto.n_trials_finalizados == 2

    reanudado = ejecutar_estudio(**_kwargs(tmp_path, "partida", n_trials=6))
    final = AlmacenManifiestosFs(tmp_path).cargar("partida")
    assert final.estado is EstadoCorrida.COMPLETADA

    def _clave(resultado):
        return sorted(
            (tuple(sorted(t.configuracion.items())), t.estado, t.valor)
            for t in resultado.trials
            if t.motivo != "configuracion_invalida"
        )

    assert _clave(reanudado) == _clave(continuo)
