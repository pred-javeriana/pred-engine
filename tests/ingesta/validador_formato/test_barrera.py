"""CSVs falsos: negativos, fechas rotas, lead time 0, duplicados."""

from __future__ import annotations

import pandas as pd
import pytest

from pred_engine.ingesta.validador_formato import (
    SchemaBarrierError,
    validate_aligned_frame,
)


def _ok(**kwargs: str) -> pd.DataFrame:
    base = {
        "sku_id": "105",
        "timestamp": "2024-10-01",
        "demand_qty": "108",
        "lead_time_days": "17",
    }
    base.update(kwargs)
    return pd.DataFrame([base])


def test_baseline_tipa_datetime64_y_no_negativos() -> None:
    marco = validate_aligned_frame(_ok())
    assert str(marco["timestamp"].dtype) == "datetime64[ns]"
    assert marco.iloc[0]["demand_qty"] == 108.0
    assert marco.iloc[0]["lead_time_days"] == 17
    assert marco.iloc[0]["sku_id"] == "105"


def test_fail_fast_demanda_negativa() -> None:
    with pytest.raises(SchemaBarrierError, match="negativa") as captured:
        validate_aligned_frame(_ok(demand_qty="-3"))
    assert captured.value.column == "demand_qty"


def test_fail_fast_fecha_invalida() -> None:
    with pytest.raises(SchemaBarrierError, match="timestamp invalido"):
        validate_aligned_frame(_ok(timestamp="not-a-date"))


def test_fail_fast_lead_time_cero() -> None:
    with pytest.raises(SchemaBarrierError, match="< 1"):
        validate_aligned_frame(_ok(lead_time_days="0"))


def test_fail_fast_duplicados() -> None:
    marco = pd.concat([_ok(), _ok()], ignore_index=True)
    with pytest.raises(SchemaBarrierError, match="duplicadas"):
        validate_aligned_frame(marco)


def test_cero_de_demanda_es_legal() -> None:
    marco = validate_aligned_frame(_ok(demand_qty="0"))
    assert marco.iloc[0]["demand_qty"] == 0.0
