"""Las entradas posteriores a la importacion no inventan otra reserva."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pred_engine.comun.metadatos_parquet import leer_metadatos_parquet
from pred_engine.ingesta import informe
from pred_engine.ingesta.categorizacion import classify_daily_panel, classify_panel
from pred_engine.ingesta.pipeline import ingest_canonical_csv
from pred_engine.ingesta.salida import publish_classified_panel


@pytest.fixture
def importacion(tmp_path: Path):
    fechas = pd.date_range("2024-01-01", periods=125)
    # El producto excluido prolonga el calendario hasta el dia 125: t*=dia 100.
    panel = pd.DataFrame(
        [
            {
                "sku_id": "ADMITIDO",
                "timestamp": fecha,
                "demand_qty": 10.0,
                "lead_time_days": 3,
            }
            for fecha in fechas[:80]
        ]
        + [
            {
                "sku_id": "TARDIO",
                "timestamp": fecha,
                "demand_qty": 0.0 if i < 100 else 5.0,
                "lead_time_days": 3,
            }
            for i, fecha in enumerate(fechas)
        ]
    )
    csv = tmp_path / "panel.csv"
    panel.to_csv(csv, index=False)
    return csv, ingest_canonical_csv(csv, data_root=tmp_path / "importado")


@pytest.mark.parametrize("entrada", ["clasificador", "alias", "publicador"])
@pytest.mark.parametrize("explicita", [False, True], ids=["omitida", "nula"])
def test_entrada_sin_reserva_se_rechaza_sin_publicar(
    tmp_path: Path, importacion, entrada: str, explicita: bool
) -> None:
    _, resultado = importacion
    diario = resultado.topology.frame.drop(columns="sku_class")
    destino = tmp_path / "salida" / "processed" / "panel.parquet"
    opciones: dict[str, Any] = {}
    if explicita:
        opciones["reserve" if entrada == "publicador" else "corte"] = None
    with pytest.raises((TypeError, ValueError)):
        if entrada == "publicador":
            publish_classified_panel(
                resultado.topology.frame,
                destino,
                data_root=tmp_path / "salida",
                daily_panel=diario,
                **opciones,
            )
        else:
            clasificador = (
                classify_panel if entrada == "clasificador" else classify_daily_panel
            )
            clasificador(diario, **opciones)
    assert not destino.parent.exists()


def test_republicacion_conserva_corte_y_exclusiones_de_importacion(
    tmp_path: Path, importacion
) -> None:
    _, resultado = importacion
    original = leer_metadatos_parquet(resultado.parquet_path)
    diario = resultado.topology.frame.drop(columns="sku_class")
    topologia = classify_daily_panel(diario, corte=original.reserve)
    (metrica,) = topologia.metrics
    assert metrica.n_periods == 80
    destino = tmp_path / "salida" / "processed" / "panel.parquet"
    publish_classified_panel(
        topologia.frame,
        destino,
        data_root=tmp_path / "salida",
        daily_panel=diario,
        reserve=original.reserve,
        exclusions=original.exclusions,
    )
    assert leer_metadatos_parquet(destino) == original
    assert original.t_star == pd.Timestamp("2024-04-09")
    assert original.reserved_days == 25
    assert [e.sku_id for e in original.exclusions] == ["TARDIO"]


def test_informe_rechaza_artefacto_sin_reserva(
    tmp_path: Path, importacion, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv, resultado = importacion
    incompleto = replace(resultado, topology=replace(resultado.topology, reserve=None))
    monkeypatch.setattr(informe, "ingest_canonical_csv", lambda *a, **kw: incompleto)
    raiz = tmp_path / "informe"
    with pytest.raises(ValueError, match="reserva"):
        informe.ingest_csv_with_report(csv, data_root=raiz)
    assert not list((raiz / "ingestas").glob("*.json"))
