"""Contrato 1.4: cinco columnas, etiquetas permitidas, constante por SKU."""

from __future__ import annotations

import sys
import types
from datetime import datetime

import pandas as pd
import pytest

from pred_engine.comun.modelos import HANDOFF_FIELDS, SKU_CLASS_LABELS
from pred_engine.ingesta.contrato_final import (
    HandoffContractError,
    default_classify_daily_panel,
    enforce_handoff_contract,
    require_positive_demand,
)


def _panel(*, etiqueta: object = "Smooth") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": pd.Series(["105", "105"], dtype="string"),
            "timestamp": pd.to_datetime([datetime(2024, 10, 1), datetime(2024, 10, 2)]),
            "demand_qty": [108.0, 0.0],
            "lead_time_days": [17, 17],
            "sku_class": [etiqueta, etiqueta],
        }
    )


def test_acepta_panel_de_cinco_columnas() -> None:
    publicado = enforce_handoff_contract(_panel())
    assert list(publicado.columns) == list(HANDOFF_FIELDS)
    assert set(publicado["sku_class"].astype(str)) <= SKU_CLASS_LABELS
    assert publicado["sku_class"].nunique() == 1


def test_rechaza_cuatro_columnas_sin_sku_class() -> None:
    marco = _panel().drop(columns=["sku_class"])
    with pytest.raises(HandoffContractError, match="exactamente"):
        enforce_handoff_contract(marco)


def test_rechaza_etiqueta_fuera_de_contrato() -> None:
    with pytest.raises(HandoffContractError, match="no permitidas"):
        enforce_handoff_contract(_panel(etiqueta="Seasonal"))


def test_rechaza_etiqueta_variable_dentro_del_sku() -> None:
    marco = _panel()
    marco.loc[1, "sku_class"] = "Lumpy"
    with pytest.raises(HandoffContractError, match="constante"):
        enforce_handoff_contract(marco)


def test_rechaza_sku_class_nulo() -> None:
    marco = _panel()
    marco.loc[1, "sku_class"] = None
    with pytest.raises(HandoffContractError, match="nulo"):
        enforce_handoff_contract(marco)


def test_rechaza_panel_vacio() -> None:
    with pytest.raises(HandoffContractError, match="no tiene filas"):
        enforce_handoff_contract(_panel().iloc[0:0])


def test_rechaza_clasificacion_que_altera_el_panel() -> None:
    diario = _panel().drop(columns=["sku_class"])
    clasificado = _panel()
    clasificado.loc[0, "demand_qty"] = 999.0
    with pytest.raises(HandoffContractError, match="mismas filas"):
        enforce_handoff_contract(clasificado, source_panel=diario)


def test_rechaza_demanda_negativa_en_handoff() -> None:
    marco = _panel()
    marco.loc[0, "demand_qty"] = -1.0
    with pytest.raises(HandoffContractError, match="negativa"):
        enforce_handoff_contract(marco)


def test_acepta_sku_con_al_menos_un_periodo_positivo() -> None:
    diario = _panel().drop(columns=["sku_class"])
    assert require_positive_demand(diario) is diario


def test_rechaza_sku_sin_demanda_positiva() -> None:
    diario = _panel().drop(columns=["sku_class"])
    diario["demand_qty"] = 0.0
    with pytest.raises(HandoffContractError, match="demanda > 0"):
        require_positive_demand(diario)


def test_rechaza_solo_el_sku_sin_demanda_positiva() -> None:
    diario = pd.DataFrame(
        {
            "sku_id": pd.Series(["105", "200"], dtype="string"),
            "timestamp": pd.to_datetime([datetime(2024, 10, 1), datetime(2024, 10, 1)]),
            "demand_qty": [108.0, 0.0],
            "lead_time_days": [17, 5],
        }
    )
    with pytest.raises(HandoffContractError, match="200") as captured:
        require_positive_demand(diario)
    assert "105" not in str(captured.value)


def test_default_classify_delega_en_1_3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _classify(panel: pd.DataFrame) -> pd.DataFrame:
        clasificado = panel.copy()
        clasificado["sku_class"] = "Erratic"
        return clasificado

    monkeypatch.setattr(
        "pred_engine.ingesta.categorizacion.classify_daily_panel",
        _classify,
        raising=False,
    )
    diario = _panel().drop(columns=["sku_class"])
    clasificado = default_classify_daily_panel(diario)
    assert list(clasificado["sku_class"]) == ["Erratic", "Erratic"]


def test_default_classify_falla_si_1_3_no_exporta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    falso = types.ModuleType("pred_engine.ingesta.categorizacion")
    monkeypatch.setitem(sys.modules, "pred_engine.ingesta.categorizacion", falso)
    with pytest.raises(HandoffContractError, match="no esta exportado"):
        default_classify_daily_panel(_panel().drop(columns=["sku_class"]))
