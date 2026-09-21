"""El DSL del espacio de busqueda: rangos, escala log, categoricos, condiciones."""

from __future__ import annotations

import numpy as np
import pytest

from pred_engine.optimizacion.optimizadores.HPO.errores import EspacioInvalidoError
from pred_engine.optimizacion.optimizadores.HPO.espacio import (
    Categorico,
    Condicion,
    Entero,
    EspacioBusqueda,
    Flotante,
    Ordinal,
    nivel_de,
)


def test_entero_muestrea_dentro_de_rango():
    rng = np.random.default_rng(0)
    parametro = Entero("p", 0, 3)
    valores = {parametro.muestrear(rng) for _ in range(200)}
    assert valores <= {0, 1, 2, 3}
    assert len(valores) > 1


def test_entero_respeta_paso():
    rng = np.random.default_rng(0)
    parametro = Entero("p", 0, 10, paso=5)
    valores = {parametro.muestrear(rng) for _ in range(200)}
    assert valores <= {0, 5, 10}


def test_entero_bajo_mayor_a_alto_es_invalido():
    with pytest.raises(EspacioInvalidoError):
        Entero("p", 5, 0)


def test_flotante_log_distribuye_en_escala_log():
    rng = np.random.default_rng(0)
    parametro = Flotante("lr", 1e-4, 1e-1, log=True)
    muestras = np.array([parametro.muestrear(rng) for _ in range(500)])
    assert muestras.min() >= 1e-4
    assert muestras.max() <= 1e-1
    log_muestras = np.log10(muestras)
    assert log_muestras.std() > 0.3


def test_flotante_log_exige_bajo_positivo():
    with pytest.raises(EspacioInvalidoError):
        Flotante("x", 0.0, 1.0, log=True)


def test_categorico_cubre_todas_las_opciones():
    rng = np.random.default_rng(0)
    parametro = Categorico("familia", ("a", "b", "c"))
    vistos = {parametro.muestrear(rng) for _ in range(200)}
    assert vistos == {"a", "b", "c"}


def test_categorico_opciones_vacias_es_invalido():
    with pytest.raises(EspacioInvalidoError):
        Categorico("x", ())


def test_ordinal_muestrea_indices_dentro_de_rango():
    rng = np.random.default_rng(0)
    parametro = Ordinal("nivel", ("bajo", "medio", "alto"))
    indices = {parametro.muestrear(rng) for _ in range(200)}
    assert indices <= {0, 1, 2}
    assert len(indices) > 1


def test_ordinal_nivel_de_decodifica_el_indice():
    parametro = Ordinal("nivel", ("bajo", "medio", "alto"))
    assert nivel_de(parametro, 0) == "bajo"
    assert nivel_de(parametro, 2) == "alto"


def test_ordinal_requiere_al_menos_dos_niveles():
    with pytest.raises(EspacioInvalidoError):
        Ordinal("nivel", ("unico",))


def test_ordinal_aparece_en_descripcion_canonica():
    espacio = EspacioBusqueda(parametros=(Ordinal("nivel", ("bajo", "alto")),))
    descripcion = espacio.descripcion_canonica()
    assert descripcion["parametros"] == [
        {"tipo": "ordinal", "nombre": "nivel", "niveles": ["bajo", "alto"]}
    ]


def test_restriccion_rechaza_configuracion_degenerada():
    espacio = EspacioBusqueda(
        parametros=(Entero("p", 0, 1), Entero("q", 0, 1)),
        restricciones=(lambda c: not (c["p"] == 0 and c["q"] == 0),),
    )
    rng = np.random.default_rng(0)
    for _ in range(100):
        configuracion = espacio.muestrear(rng)
        assert not (configuracion["p"] == 0 and configuracion["q"] == 0)


def test_condicion_activa_solo_con_el_valor_del_padre():
    espacio = EspacioBusqueda(
        parametros=(
            Categorico("n_capas", (1, 2)),
            Entero("unidades_capa_2", 8, 64),
        ),
        condiciones=(
            Condicion(parametro="unidades_capa_2", padre="n_capas", valores=(2,)),
        ),
    )
    rng = np.random.default_rng(0)
    muestras = [espacio.muestrear(rng) for _ in range(300)]
    con_una_capa = [c for c in muestras if c["n_capas"] == 1]
    con_dos_capas = [c for c in muestras if c["n_capas"] == 2]
    assert con_una_capa and con_dos_capas
    assert all("unidades_capa_2" not in c for c in con_una_capa)
    assert all("unidades_capa_2" in c for c in con_dos_capas)


def test_nombres_activos_refleja_la_condicion():
    espacio = EspacioBusqueda(
        parametros=(Categorico("modo", ("a", "b")), Flotante("solo_b", 0.0, 1.0)),
        condiciones=(Condicion(parametro="solo_b", padre="modo", valores=("b",)),),
    )
    assert espacio.nombres_activos({"modo": "a"}) == ("modo",)
    assert espacio.nombres_activos({"modo": "b", "solo_b": 0.5}) == ("modo", "solo_b")


def test_nombres_duplicados_es_invalido():
    with pytest.raises(EspacioInvalidoError):
        EspacioBusqueda(parametros=(Entero("p", 0, 1), Entero("p", 0, 2)))


def test_condicion_con_padre_inexistente_es_invalida():
    with pytest.raises(EspacioInvalidoError):
        EspacioBusqueda(
            parametros=(Entero("p", 0, 1),),
            condiciones=(Condicion(parametro="p", padre="fantasma", valores=(1,)),),
        )


def test_condicion_con_padre_declarado_despues_es_invalida():
    with pytest.raises(EspacioInvalidoError, match="antes"):
        EspacioBusqueda(
            parametros=(Entero("hijo", 0, 1), Categorico("padre", (1, 2))),
            condiciones=(Condicion(parametro="hijo", padre="padre", valores=(1,)),),
        )


def test_muestreo_imposible_lanza_error_claro():
    espacio = EspacioBusqueda(
        parametros=(Entero("p", 0, 1),),
        restricciones=(lambda c: False,),
    )
    with pytest.raises(EspacioInvalidoError):
        espacio.muestrear(np.random.default_rng(0), intentos_max=10)


def test_descripcion_canonica_es_determinista_e_independiente_de_callables_anonimos():
    def _cota(configuracion):
        return configuracion["p"] < 3

    a = EspacioBusqueda(parametros=(Entero("p", 0, 5),), restricciones=(_cota,))
    b = EspacioBusqueda(parametros=(Entero("p", 0, 5),), restricciones=(_cota,))
    assert a.descripcion_canonica() == b.descripcion_canonica()


def test_descripcion_canonica_cambia_si_cambia_el_rango():
    a = EspacioBusqueda(parametros=(Entero("p", 0, 5),))
    b = EspacioBusqueda(parametros=(Entero("p", 0, 6),))
    assert a.descripcion_canonica() != b.descripcion_canonica()
