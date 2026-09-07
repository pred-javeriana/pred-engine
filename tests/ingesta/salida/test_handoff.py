"""Handoff 1.2 → 1.3 → 1.4 con clasificador real (sin stub)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pred_engine.comun.modelos import PANEL_FIELDS
from pred_engine.ingesta.pipeline import run_classify_csv, run_verify_parquet
from pred_engine.ingesta.salida import HandoffPreconditionError

_CSV_OFICIAL = Path("local_data") / "inventory_data.csv"


def test_salida_no_duplica_nucleo_sbc() -> None:
    raiz = Path("src") / "pred_engine" / "ingesta" / "salida"
    for archivo in raiz.glob("*.py"):
        texto = archivo.read_text(encoding="utf-8")
        assert "compute_adi" not in texto
        assert "compute_cv2" not in texto
        assert "route_syntetos_boylan" not in texto
        assert "categorizacion" not in texto


def test_sku_sin_demanda_positiva_no_persiste(tmp_path: Path) -> None:
    csv = tmp_path / "ceros.csv"
    csv.write_text(
        "sku_id,timestamp,demand_qty,lead_time_days\n105,2024-10-01,0,17\n",
        encoding="utf-8",
    )
    raiz = tmp_path / "data"
    with pytest.raises(HandoffPreconditionError, match="sin demanda"):
        run_classify_csv(csv, data_root=raiz)
    procesados = list((raiz / "processed").glob("*.parquet"))
    assert procesados == []


def test_flujo_canonico_mini_csv(tmp_path: Path) -> None:
    csv = tmp_path / "mini.csv"
    csv.write_text(
        "sku_id,timestamp,demand_qty,lead_time_days,Item_Name\n"
        "105,2024-10-01,108,17,Ventilator\n"
        "105,2024-10-04,50,17,Ventilator\n",
        encoding="utf-8",
    )
    topologia, destino = run_classify_csv(csv, data_root=tmp_path / "data")
    leido = run_verify_parquet(destino)
    assert list(leido.columns) == list(PANEL_FIELDS)
    assert len(leido) == 4
    assert set(leido["sku_class"].astype(str)) == {"intermittent"}
    assert topologia.metrics[0].sku_class == "intermittent"
    assert pd.api.types.is_datetime64_any_dtype(leido["timestamp"])
    assert pd.api.types.is_float_dtype(leido["demand_qty"])
    assert pd.api.types.is_integer_dtype(leido["lead_time_days"])


@pytest.mark.skipif(not _CSV_OFICIAL.is_file(), reason="dataset oficial ausente")
def test_handoff_dataset_oficial(tmp_path: Path) -> None:
    topologia, destino = run_classify_csv(_CSV_OFICIAL, data_root=tmp_path / "data")
    leido = run_verify_parquet(destino)
    assert len(leido) == 4862
    assert int(leido["sku_id"].nunique()) == 10
    etiquetas = leido.drop_duplicates("sku_id")["sku_class"].astype(str).tolist()
    assert etiquetas == ["intermittent"] * 10
    assert {m.sku_class for m in topologia.metrics} == {"intermittent"}
    assert len(topologia.metrics) == 10
    nunique = leido.groupby("sku_id")["sku_class"].nunique()
    assert bool((nunique == 1).all())
