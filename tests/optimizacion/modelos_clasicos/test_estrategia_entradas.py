"""La estrategia clasica rechaza solicitudes invalidas con datos reales del contrato."""

from datetime import datetime

import pytest

from pred_engine.comun.modelos import ClassifiedObservation
from pred_engine.optimizacion.optimizadores.modelos_clasicos.estrategia import (
    ClassicalSelectionStrategy,
    PresupuestoClasico,
)
from pred_engine.optimizacion.router import SelectionContractError, SelectionRequest


def test_persistencia_clasica_exige_sesion(tmp_path):
    with pytest.raises(ValueError, match="sesion es obligatoria"):
        ClassicalSelectionStrategy(raiz_corrida=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_perfil_sin_presupuesto_se_rechaza():
    estrategia = ClassicalSelectionStrategy(
        presupuestos={"dense_stable": PresupuestoClasico(n_trials=1)}
    )
    solicitud = SelectionRequest(sku_id="S1", sku_class="erratic")
    with pytest.raises(SelectionContractError, match="sin presupuesto"):
        estrategia.select(solicitud, "dense_variable")


def test_solicitud_sin_historia_se_rechaza():
    solicitud = SelectionRequest(sku_id="S1", sku_class="smooth")
    with pytest.raises(SelectionContractError, match="no trae serie"):
        ClassicalSelectionStrategy().select(solicitud, "dense_stable")


def test_historia_corta_conserva_la_causa_del_error():
    solicitud = SelectionRequest(
        sku_id="S1",
        sku_class="smooth",
        series=(
            ClassifiedObservation(
                sku_id="S1",
                sku_class="smooth",
                timestamp=datetime(2024, 1, 1),
                demand_qty=5.0,
                lead_time_days=3,
            ),
        ),
    )
    with pytest.raises(SelectionContractError, match="insuficiente") as error:
        ClassicalSelectionStrategy().select(solicitud, "dense_stable")
    assert isinstance(error.value.__cause__, ValueError)
