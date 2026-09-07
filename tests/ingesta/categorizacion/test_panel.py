"""sku_class constante por SKU; columnas transaccionales intactas."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from pred_engine.comun.modelos import PANEL_FIELDS
from pred_engine.ingesta.categorizacion import (
    TopologyContractError,
    classify_daily_panel,
    classify_panel,
)


def _panel_cuatro_clases() -> pd.DataFrame:
    # Smooth: 4 dias densos constantes.
    suave = pd.DataFrame(
        {
            "sku_id": ["S"] * 4,
            "timestamp": pd.date_range(datetime(2024, 10, 1), periods=4, freq="D"),
            "demand_qty": [5.0, 5.0, 5.0, 5.0],
            "lead_time_days": [3, 3, 3, 3],
        }
    )
    # Intermittent: ADI=3.5, CV2=0.
    inter = pd.DataFrame(
        {
            "sku_id": ["I"] * 7,
            "timestamp": pd.date_range(datetime(2024, 10, 1), periods=7, freq="D"),
            "demand_qty": [5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
            "lead_time_days": [3] * 7,
        }
    )
    # Erratic: ADI=1, CV2 alto.
    err = pd.DataFrame(
        {
            "sku_id": ["E"] * 4,
            "timestamp": pd.date_range(datetime(2024, 10, 1), periods=4, freq="D"),
            "demand_qty": [1.0, 10.0, 2.0, 20.0],
            "lead_time_days": [3] * 4,
        }
    )
    # Lumpy: ADI=3, CV2 alto.
    lumpy = pd.DataFrame(
        {
            "sku_id": ["L"] * 6,
            "timestamp": pd.date_range(datetime(2024, 10, 1), periods=6, freq="D"),
            "demand_qty": [10.0, 0.0, 0.0, 100.0, 0.0, 0.0],
            "lead_time_days": [3] * 6,
        }
    )
    return pd.concat([suave, inter, err, lumpy], ignore_index=True)


def test_cuatro_cuadrantes_y_etiqueta_constante() -> None:
    origen = _panel_cuatro_clases()
    copia = origen.copy()
    artefacto = classify_panel(origen)
    assert origen.equals(copia)
    assert list(artefacto.frame.columns) == list(PANEL_FIELDS)
    por_sku = artefacto.frame.groupby("sku_id")["sku_class"].nunique().to_dict()
    assert por_sku == {"E": 1, "I": 1, "L": 1, "S": 1}
    etiquetas = (
        artefacto.frame.drop_duplicates("sku_id")
        .set_index("sku_id")["sku_class"]
        .to_dict()
    )
    assert etiquetas == {
        "S": "smooth",
        "I": "intermittent",
        "E": "erratic",
        "L": "lumpy",
    }
    # Inmutabilidad transaccional: demanda del SKU S intacta.
    original_s = origen.loc[origen["sku_id"] == "S", "demand_qty"].tolist()
    nuevo_s = artefacto.frame.loc[
        artefacto.frame["sku_id"] == "S", "demand_qty"
    ].tolist()
    assert original_s == nuevo_s


def test_panel_vacio_lanza() -> None:
    with pytest.raises(TopologyContractError):
        classify_panel(pd.DataFrame())


def test_classify_daily_panel_es_el_clasificador_real() -> None:
    origen = _panel_cuatro_clases()
    a = classify_panel(origen)
    b = classify_daily_panel(origen)
    assert a.frame.equals(b.frame)
    assert [m.sku_class for m in a.metrics] == [m.sku_class for m in b.metrics]
