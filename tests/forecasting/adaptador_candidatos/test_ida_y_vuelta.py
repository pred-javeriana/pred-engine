"""Ida y vuelta M2 -> M3 (mitigacion de ADR-03-004): lo que M2 serializa, M3 lo
valida sin fallos y lo instancia. Corre las estrategias reales con presupuesto
minimo, asi una divergencia del esquema compartido rompe esta prueba."""

from __future__ import annotations

import numpy as np
import pytest
from tests._dobles import PipelineFalso
from tests._manifiestos import CONTEXTO, EMITIDO_EN
from tests.optimizacion.modelos_deep_learning.test_dl_selection import CHICO
from tests.optimizacion.modelos_machine_learning.test_estrategia import (
    _CHICO,
    _solicitud,
)

from pred_engine.forecasting.adaptador_candidatos import (
    instanciar,
    validar_manifiesto,
)
from pred_engine.optimizacion.optimizadores.modelos_clasicos import EspacioClasico
from pred_engine.optimizacion.optimizadores.modelos_clasicos.estrategia import (
    ClassicalSelectionStrategy,
    PresupuestoClasico,
)
from pred_engine.optimizacion.optimizadores.modelos_deep_learning import (
    DLSelectionStrategy,
)
from pred_engine.optimizacion.optimizadores.modelos_fundacionales.estrategia import (
    FoundationSelectionStrategy,
)
from pred_engine.optimizacion.optimizadores.modelos_machine_learning import (
    MLSelectionStrategy,
    PresupuestoHPO,
)
from pred_engine.optimizacion.router import construir_manifiesto

pytestmark = pytest.mark.slow


def _resultados_m2():
    solicitud = _solicitud()
    estrategias = (
        ClassicalSelectionStrategy(
            espacio=EspacioClasico(
                p_max=1, d_max=1, q_max=1, P_max=0, D_max=0, Q_max=0, m=7
            ),
            presupuestos={"dense_stable": PresupuestoClasico(n_trials=2)},
        ),
        MLSelectionStrategy(
            espacio=_CHICO,
            presupuestos={"dense_stable": PresupuestoHPO(n_trials=2)},
        ),
        DLSelectionStrategy(
            espacio=CHICO, n_trials=2, min_train=10, horizonte=2, paso=5
        ),
        FoundationSelectionStrategy(),
    )
    return [e.select(solicitud, "dense_stable") for e in estrategias]


def _handoff(resultados):
    contenido = construir_manifiesto(
        resultados,
        run_id_m2="m2-ida-y-vuelta",
        contexto=CONTEXTO,
        emitido_en=EMITIDO_EN,
    )
    return validar_manifiesto(
        contenido,
        esperado=CONTEXTO,
        skus_panel={"S1"},
        cargar_fundacional=lambda configuracion: PipelineFalso(),
    )


def test_manifiesto_de_m2_se_valida_sin_fallos_e_instancia_en_m3() -> None:
    handoff = _handoff(_resultados_m2())

    assert handoff.fallos == ()
    serie = np.clip(np.arange(60, dtype=float) % 7 + 10, 0, None)
    for candidato in handoff.candidatos:
        modelo = instanciar(candidato, pipeline_fundacional=PipelineFalso())
        assert modelo.fit(serie).predict(3).shape == (3,)


def test_solo_pasa_a_m3_el_ganador_del_hpo_de_cada_familia() -> None:
    resultados = _resultados_m2()
    handoff = _handoff(resultados)

    # El HPO probo varios trials por familia, pero pasa uno por familia y SKU.
    assert all(r.payload["n_trials"] > 1 for r in resultados[:3])
    assert [c.familia for c in handoff.candidatos] == [
        "classical",
        "ml",
        "dl",
        "foundation",
    ]
    clasico, ml, dl, _ = handoff.candidatos
    # Y es exactamente la configuracion ganadora que M2 registro como evidencia.
    ganador_clasico = resultados[0].payload
    configuracion = clasico.configuracion.model_dump()
    assert [configuracion[k] for k in "pdq"] == ganador_clasico["order"]
    assert [configuracion[k] for k in "PDQm"] == ganador_clasico["seasonal_order"]
    assert (
        ml.configuracion.model_dump().items()
        >= resultados[1].payload["hiperparametros"].items()
    )
    assert dl.configuracion.model_dump() == resultados[2].payload["hiperparametros"]
