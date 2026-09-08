"""Preservacion: 1.3 solo puede anadir sku_class."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from pred_engine.ingesta.categorizacion import classify_daily_panel
from pred_engine.ingesta.salida import PanelPreservationError, require_panel_preserved


def _diario_mini() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": pd.Series(["105"] * 4, dtype="string"),
            "timestamp": pd.date_range(datetime(2024, 10, 1), periods=4, freq="D"),
            "demand_qty": [108.0, 0.0, 0.0, 50.0],
            "lead_time_days": pd.Series([17, 17, 17, 17], dtype="int64"),
        }
    )


def test_clasificador_real_preserva_panel() -> None:
    diario = _diario_mini()
    copia = diario.copy()
    clasificado = classify_daily_panel(diario).frame
    require_panel_preserved(diario, clasificado)
    assert diario.equals(copia)


def test_rechaza_reordenamiento() -> None:
    diario = _diario_mini()
    clasificado = classify_daily_panel(diario).frame.iloc[::-1].reset_index(drop=True)
    with pytest.raises(PanelPreservationError, match="reorden"):
        require_panel_preserved(diario, clasificado)


def test_rechaza_eliminacion_de_filas() -> None:
    diario = _diario_mini()
    clasificado = classify_daily_panel(diario).frame.iloc[:-1]
    with pytest.raises(PanelPreservationError, match="filas"):
        require_panel_preserved(diario, clasificado)


def test_rechaza_mutacion_de_demanda() -> None:
    diario = _diario_mini()
    clasificado = classify_daily_panel(diario).frame.copy()
    clasificado.loc[clasificado.index[0], "demand_qty"] = 999.0
    with pytest.raises(PanelPreservationError, match="demand_qty"):
        require_panel_preserved(diario, clasificado)


def test_rechaza_columna_extra() -> None:
    diario = _diario_mini()
    clasificado = classify_daily_panel(diario).frame.copy()
    clasificado["adi"] = 1.32
    with pytest.raises(PanelPreservationError, match="no permitidas"):
        require_panel_preserved(diario, clasificado)
