"""Validador 1.4: contrato exacto, fail-closed, sin mutar el marco."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from pred_engine.comun.modelos import PANEL_FIELDS
from pred_engine.ingesta.categorizacion import classify_daily_panel
from pred_engine.ingesta.salida import OutputContractError, validate_output_contract


def _diario_mini() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": pd.Series(["105"] * 4, dtype="string"),
            "timestamp": pd.date_range(datetime(2024, 10, 1), periods=4, freq="D"),
            "demand_qty": [108.0, 0.0, 0.0, 50.0],
            "lead_time_days": pd.Series([17, 17, 17, 17], dtype="int64"),
        }
    )


def _clasificado_valido() -> pd.DataFrame:
    return classify_daily_panel(_diario_mini()).frame


def test_acepta_panel_clasificado_canonico() -> None:
    marco = _clasificado_valido()
    copia = marco.copy()
    validate_output_contract(marco)
    assert marco.equals(copia)
    assert list(marco.columns) == list(PANEL_FIELDS)


def test_rechaza_columnas_extra() -> None:
    marco = _clasificado_valido()
    marco = marco.copy()
    marco["extra"] = 1.0
    with pytest.raises(OutputContractError, match="exactamente"):
        validate_output_contract(marco)


def test_rechaza_orden_incorrecto() -> None:
    marco = _clasificado_valido()[["sku_class", *PANEL_FIELDS[:-1]]]
    with pytest.raises(OutputContractError, match="exactamente"):
        validate_output_contract(marco)


def test_rechaza_nulos_en_obligatorias() -> None:
    marco = _clasificado_valido().copy()
    marco.loc[marco.index[0], "timestamp"] = pd.NaT
    with pytest.raises(OutputContractError, match="nulos"):
        validate_output_contract(marco)


def test_rechaza_demanda_negativa() -> None:
    marco = _clasificado_valido().copy()
    marco.loc[marco.index[0], "demand_qty"] = -0.1
    with pytest.raises(OutputContractError, match="negativos"):
        validate_output_contract(marco)


def test_rechaza_lead_time_fuera_de_dominio() -> None:
    marco = _clasificado_valido().copy()
    marco.loc[marco.index[0], "lead_time_days"] = 0
    with pytest.raises(OutputContractError, match="dominio"):
        validate_output_contract(marco)


def test_rechaza_sku_class_fuera_del_conjunto() -> None:
    marco = _clasificado_valido().copy()
    marco["sku_class"] = pd.Series(["intermitente"] * len(marco), dtype="string")
    with pytest.raises(OutputContractError, match="sku_class"):
        validate_output_contract(marco)


def test_rechaza_sku_class_no_constante() -> None:
    marco = _clasificado_valido().copy()
    marco.loc[marco.index[0], "sku_class"] = "smooth"
    with pytest.raises(OutputContractError, match="constante"):
        validate_output_contract(marco)


def test_rechaza_demand_qty_no_float() -> None:
    marco = _clasificado_valido().copy()
    marco["demand_qty"] = marco["demand_qty"].astype("int64")
    with pytest.raises(OutputContractError, match="Float"):
        validate_output_contract(marco)
