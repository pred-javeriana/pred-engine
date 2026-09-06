"""Pipeline de ingesta: sonda 1.2, contrato 1.4, sin rename automatico."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from pred_engine.comun.modelos import CANONICAL_FIELDS, HANDOFF_FIELDS, SKU_CLASS_LABELS
from pred_engine.ingesta.contrato_final import HandoffContractError
from pred_engine.ingesta.pipeline import run_ingest
from pred_engine.ingesta.sonda import SemanticAlignmentError


class FakeLlmProvider:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def complete(self, prompt: str, *, temperature: float, timeout: float) -> str:
        return json.dumps(self.payload)


_ACCEPTED = {
    "status": "accepted",
    "diagnostic": [
        {
            "field": "schema",
            "severity": "info",
            "message": "Cabeceras canonicas presentes",
            "action": None,
        }
    ],
}

_REJECTED = {
    "status": "rejected",
    "diagnostic": [
        {
            "field": "timestamp",
            "severity": "error",
            "message": "Renombrar Date",
            "action": "Renombrar 'Date' a 'timestamp'",
        }
    ],
}


def _classify_smooth(panel: pd.DataFrame) -> pd.DataFrame:
    clasificado = panel.copy()
    clasificado["sku_class"] = "Smooth"
    return clasificado


def _classify_por_sku(panel: pd.DataFrame) -> pd.DataFrame:
    clasificado = panel.copy()
    mapa = {"105": "Lumpy", "200": "Smooth"}
    clasificado["sku_class"] = clasificado["sku_id"].map(mapa)
    return clasificado


def test_run_ingest_csv_canonico_con_huecos(tmp_path: Path) -> None:
    csv = tmp_path / "mini.csv"
    csv.write_text(
        "sku_id,timestamp,demand_qty,lead_time_days\n"
        "105,2024-10-01,108,17\n"
        "105,2024-10-04,50,17\n",
        encoding="utf-8",
    )
    raiz = tmp_path / "data"
    resultado = run_ingest(
        csv,
        FakeLlmProvider(_ACCEPTED),
        data_root=raiz,
        timeout=5.0,
        classify=_classify_smooth,
    )
    assert resultado.parquet_path.is_file()
    assert resultado.parquet_path == raiz / "processed" / "mini.parquet"
    assert resultado.diagnostic.diagnostic.is_accepted()
    assert list(resultado.diagnostic.frame.columns) == list(CANONICAL_FIELDS)
    assert list(resultado.validated.columns) == list(CANONICAL_FIELDS)
    assert list(resultado.panel.columns) == list(CANONICAL_FIELDS)
    publicado = pd.read_parquet(resultado.parquet_path)
    assert list(publicado.columns) == list(HANDOFF_FIELDS)
    assert len(publicado) == 4
    assert list(publicado["demand_qty"]) == [108.0, 0.0, 0.0, 50.0]
    assert set(publicado["sku_class"].astype(str)) <= SKU_CLASS_LABELS
    assert publicado["sku_class"].nunique() == 1
    esquema = pq.read_schema(resultado.parquet_path)
    assert list(esquema.names) == list(HANDOFF_FIELDS)


def test_run_ingest_sku_class_constante_por_sku(tmp_path: Path) -> None:
    csv = tmp_path / "dos_sku.csv"
    csv.write_text(
        "sku_id,timestamp,demand_qty,lead_time_days\n"
        "105,2024-10-01,108,17\n"
        "105,2024-10-04,50,17\n"
        "200,2024-10-01,10,5\n"
        "200,2024-10-02,12,5\n",
        encoding="utf-8",
    )
    raiz = tmp_path / "data"
    resultado = run_ingest(
        csv,
        FakeLlmProvider(_ACCEPTED),
        data_root=raiz,
        timeout=5.0,
        classify=_classify_por_sku,
    )
    publicado = pd.read_parquet(resultado.parquet_path)
    assert list(publicado.columns) == list(HANDOFF_FIELDS)
    por_sku = publicado.groupby("sku_id")["sku_class"].nunique()
    assert (por_sku == 1).all()
    sku_105 = set(publicado.loc[publicado["sku_id"] == "105", "sku_class"].astype(str))
    sku_200 = set(publicado.loc[publicado["sku_id"] == "200", "sku_class"].astype(str))
    assert sku_105 == {"Lumpy"}
    assert sku_200 == {"Smooth"}
    assert set(publicado["sku_class"].astype(str)) <= SKU_CLASS_LABELS


def test_run_ingest_rechaza_clasificacion_fuera_de_contrato(tmp_path: Path) -> None:
    csv = tmp_path / "mini.csv"
    csv.write_text(
        "sku_id,timestamp,demand_qty,lead_time_days\n105,2024-10-01,108,17\n",
        encoding="utf-8",
    )

    def _classify_invalido(panel: pd.DataFrame) -> pd.DataFrame:
        clasificado = panel.copy()
        clasificado["sku_class"] = "Seasonal"
        return clasificado

    with pytest.raises(HandoffContractError, match="no permitidas"):
        run_ingest(
            csv,
            FakeLlmProvider(_ACCEPTED),
            data_root=tmp_path / "data",
            timeout=5.0,
            classify=_classify_invalido,
        )


@pytest.mark.parametrize("alteracion", ["drop", "demand"])
def test_run_ingest_rechaza_filas_alteradas_por_clasificador(
    tmp_path: Path,
    alteracion: str,
) -> None:
    csv = tmp_path / "mini.csv"
    csv.write_text(
        "sku_id,timestamp,demand_qty,lead_time_days\n"
        "105,2024-10-01,108,17\n"
        "105,2024-10-02,0,17\n",
        encoding="utf-8",
    )

    def _classify_alterado(panel: pd.DataFrame) -> pd.DataFrame:
        if alteracion == "drop":
            panel = panel.iloc[:-1].copy()
        else:
            panel = panel.copy()
            panel.loc[0, "demand_qty"] = 109.0
        panel["sku_class"] = "Smooth"
        return panel

    raiz = tmp_path / "data"
    with pytest.raises(HandoffContractError, match="mismas filas"):
        run_ingest(
            csv,
            FakeLlmProvider(_ACCEPTED),
            data_root=raiz,
            timeout=5.0,
            classify=_classify_alterado,
        )
    assert list((raiz / "processed").glob("*.parquet")) == []


def test_run_ingest_rechaza_sku_sin_demanda_positiva_sin_clasificar(
    tmp_path: Path,
) -> None:
    csv = tmp_path / "ceros.csv"
    csv.write_text(
        "sku_id,timestamp,demand_qty,lead_time_days\n"
        "105,2024-10-01,0,17\n"
        "105,2024-10-02,0,17\n",
        encoding="utf-8",
    )
    llamadas: list[int] = []

    def _classify_no_debe_correr(panel: pd.DataFrame) -> pd.DataFrame:
        llamadas.append(1)
        clasificado = panel.copy()
        clasificado["sku_class"] = "Smooth"
        return clasificado

    raiz = tmp_path / "data"
    with pytest.raises(HandoffContractError, match="demanda > 0"):
        run_ingest(
            csv,
            FakeLlmProvider(_ACCEPTED),
            data_root=raiz,
            timeout=5.0,
            classify=_classify_no_debe_correr,
        )
    assert llamadas == []
    assert list((raiz / "processed").glob("*.parquet")) == []


def test_run_ingest_rechaza_csv_no_canonico_sin_mutar(tmp_path: Path) -> None:
    csv = tmp_path / "hostil.csv"
    original = (
        "Date,Item_ID,Avg_Usage_Per_Day,Restock_Lead_Time\n2024-10-01,105,108,17\n"
    )
    csv.write_text(original, encoding="utf-8")
    raiz = tmp_path / "data"
    with pytest.raises(SemanticAlignmentError) as exc:
        run_ingest(csv, FakeLlmProvider(_REJECTED), data_root=raiz, timeout=5.0)
    assert exc.value.diagnostic is not None
    assert exc.value.diagnostic.is_rejected()
    assert csv.read_text(encoding="utf-8") == original
    assert list((raiz / "processed").glob("*.parquet")) == []
