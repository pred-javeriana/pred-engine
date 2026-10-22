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


def _marco(filas: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        filas, columns=["sku_id", "timestamp", "demand_qty", "lead_time_days"]
    ).astype("str")


def test_formatos_validos_poco_comunes_se_tipan_igual() -> None:
    marco = validate_aligned_frame(
        _marco(
            [
                ("105", "2024-10-01", "108", "17"),
                (" 106 ", " 2024-10-02T05:06:07 ", "1_000", "١"),
                ("107", "2024-10-03", " 2.5e1 ", "3.0"),
            ]
        )
    )
    assert marco["sku_id"].tolist() == ["105", "106", "107"]
    assert marco["timestamp"].tolist() == list(
        pd.to_datetime(["2024-10-01", "2024-10-02", "2024-10-03"])
    )
    assert marco["demand_qty"].tolist() == [108.0, 1000.0, 25.0]
    assert marco["lead_time_days"].tolist() == [17, 1, 3]
    assert [str(d) for d in marco.dtypes] == [
        "string",
        "datetime64[ns]",
        "float64",
        "int64",
    ]


def test_el_error_es_el_de_la_primera_fila_invalida() -> None:
    with pytest.raises(SchemaBarrierError, match="negativa") as captured:
        validate_aligned_frame(
            _marco(
                [
                    ("105", "2024-10-01", "1", "3"),
                    ("105", "2024-10-02", " 1e1 ", "3"),
                    ("105", "2024-10-03", "-1", "3"),
                    ("105", "no-es-fecha", "1", "0"),
                ]
            )
        )
    assert (captured.value.row_index, captured.value.column) == (2, "demand_qty")
    assert captured.value.raw_value == "-1"


def test_la_primera_comprobacion_fallida_de_la_fila_decide_el_error() -> None:
    with pytest.raises(SchemaBarrierError, match="timestamp invalido") as captured:
        validate_aligned_frame(
            _marco([("105", "2024-10-01", "1", "3"), ("105", "2024-02-30", "-1", "0")])
        )
    assert (captured.value.row_index, captured.value.column) == (1, "timestamp")


def test_sku_en_blanco_es_violacion_pydantic() -> None:
    with pytest.raises(SchemaBarrierError, match="violacion Pydantic en fila 1"):
        validate_aligned_frame(
            _marco([("105", "2024-10-01", "1", "3"), ("\t ", "2024-10-02", "1", "3")])
        )


def test_lead_time_no_entero_y_demanda_no_finita() -> None:
    with pytest.raises(SchemaBarrierError, match="no entero"):
        validate_aligned_frame(_ok(lead_time_days="2.5"))
    with pytest.raises(SchemaBarrierError, match="violacion Pydantic"):
        validate_aligned_frame(_ok(demand_qty="nan"))


def test_un_marco_no_textual_se_valida_fila_a_fila() -> None:
    marco = validate_aligned_frame(
        pd.DataFrame(
            {
                "sku_id": ["105"],
                "timestamp": ["2024-10-01"],
                "demand_qty": [2.0],
                "lead_time_days": [4],
            }
        )
    )
    assert marco.iloc[0]["demand_qty"] == 2.0
    assert marco.iloc[0]["lead_time_days"] == 4
