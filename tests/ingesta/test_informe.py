"""Informe de ingesta en archivo (G1): un JSON por ingesta, tambien si falla."""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from pred_engine.ingesta.data import RawConflictError, RawWritePermissionError
from pred_engine.ingesta.informe import (
    REPORT_SCHEMA,
    IngestReportFile,
    error_code,
    ingest_csv_with_report,
    report_path,
)
from pred_engine.ingesta.salida.errores import HandoffPreconditionError
from pred_engine.ingesta.validador_formato.errores import SchemaBarrierError


def _csv(ruta: Path, *, salto: int = 0, filas_extra: list[str] | None = None) -> Path:
    inicio = date(2026, 1, 1)
    lineas = ["sku_id,timestamp,demand_qty,lead_time_days"]
    for i in range(60):
        dia = (inicio + timedelta(days=i)).isoformat()
        lineas.append(f"FIL-0001,{dia},{5 + i % 7 + salto},7")
        lineas.append(f"FIL-0002,{dia},{4 if i % 3 == 0 else 0},7")
    lineas.extend(filas_extra or [])
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta


def _sha(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _json(ruta: Path) -> dict[str, Any]:
    return json.loads(ruta.read_text(encoding="utf-8"))


class _Proveedor:
    def __init__(self, carga: dict[str, object]) -> None:
        self.carga = carga

    def complete(self, prompt: str, *, temperature: float, timeout: float) -> str:
        return json.dumps(self.carga)


# ------------------------------------------------------------------------- aceptada
def test_ingesta_aceptada_deja_el_informe_completo(tmp_path: Path) -> None:
    origen = _csv(tmp_path / "entrada" / "ventas.csv")
    raiz = tmp_path / "data"

    salida = ingest_csv_with_report(origen, data_root=raiz)

    assert salida.error is None
    assert salida.path == raiz / "ingestas" / f"{_sha(origen)}.json"
    datos = _json(salida.path)
    assert datos["schema"] == REPORT_SCHEMA
    assert datos["ingest_id"] == _sha(origen)
    assert datos["status"] == "accepted"
    assert datos["source_file"] == {
        "datos_sinteticos": None,
        "name": "ventas.csv",
        "sha256": _sha(origen),
        "row_count": 120,
    }
    assert datos["deposit"] == {
        "raw_path": "raw/ventas.csv",
        "status": "deposited",
        "would_overwrite": False,
    }
    assert datos["header_diagnostic"] is None
    assert datos["validation"] == {
        "rows_validated": 120,
        "rows_daily_panel": 120,
        "n_skus": 2,
    }
    assert datos["published"]["parquet_path"] == "processed/ventas.parquet"
    assert datos["published"]["sha256"] == _sha(raiz / "processed/ventas.parquet")
    assert datos["published"]["columns"][-1] == "sku_class"
    assert datos["error"] is None
    assert datos["quality_log"] == []


def test_el_informe_trae_reserva_y_metricas_por_sku(tmp_path: Path) -> None:
    salida = ingest_csv_with_report(
        _csv(tmp_path / "e" / "v.csv"), data_root=tmp_path / "d"
    )

    datos = _json(salida.path)
    assert datos["reserve"]["t_star"] == "2026-02-17"
    assert datos["reserve"]["reserved_days"] == 12
    assert datos["topology"]["thresholds"] == {"adi": 1.32, "cv2": 0.49}
    metricas = {m["sku_id"]: m for m in datos["topology"]["metrics"]}
    assert set(metricas) == {"FIL-0001", "FIL-0002"}
    assert metricas["FIL-0002"]["n_positive"] < metricas["FIL-0002"]["n_periods"]
    assert {m["sku_class"] for m in metricas.values()} <= {
        "smooth",
        "intermittent",
        "erratic",
        "lumpy",
    }


def test_el_informe_se_relee_con_su_modelo(tmp_path: Path) -> None:
    salida = ingest_csv_with_report(
        _csv(tmp_path / "e" / "v.csv"), data_root=tmp_path / "d"
    )
    leido = IngestReportFile.model_validate_json(
        salida.path.read_text(encoding="utf-8")
    )
    assert leido == salida.report


def test_repetir_la_ingesta_reescribe_el_mismo_informe(tmp_path: Path) -> None:
    origen = _csv(tmp_path / "e" / "v.csv")
    raiz = tmp_path / "d"
    primero = ingest_csv_with_report(origen, data_root=raiz)
    segundo = ingest_csv_with_report(origen, data_root=raiz)

    assert segundo.path == primero.path
    datos = _json(segundo.path)
    assert datos["deposit"]["status"] == "already_present_identical"
    assert datos["topology"] == _json(primero.path)["topology"]
    assert [p.name for p in (raiz / "ingestas").iterdir()] == [f"{_sha(origen)}.json"]


def test_no_quedan_temporales(tmp_path: Path) -> None:
    raiz = tmp_path / "d"
    ingest_csv_with_report(_csv(tmp_path / "e" / "v.csv"), data_root=raiz)
    assert not [p for p in (raiz / "ingestas").iterdir() if p.name.endswith(".part")]


# ------------------------------------------------------------------------ conflicto
def test_conflicto_sin_confirmar_queda_a_la_espera_y_no_publica(tmp_path: Path) -> None:
    raiz = tmp_path / "d"
    ingest_csv_with_report(_csv(tmp_path / "a" / "ventas.csv"), data_root=raiz)
    nuevo = _csv(tmp_path / "b" / "ventas.csv", salto=3)

    salida = ingest_csv_with_report(nuevo, data_root=raiz)

    assert isinstance(salida.error, RawConflictError)
    assert salida.result is None
    datos = _json(raiz / "ingestas" / f"{_sha(nuevo)}.json")
    assert datos["status"] == "needs_confirmation"
    assert datos["deposit"] == {
        "raw_path": "raw/ventas.csv",
        "status": "already_present_different",
        "would_overwrite": True,
    }
    assert datos["quality_log"][0]["code"] == "ingest.raw_conflict"
    assert datos["published"] is None and datos["topology"] is None
    assert [p.name for p in (raiz / "processed").iterdir()] == ["ventas.parquet"]


def test_conflicto_confirmado_conserva_ambos_y_acepta(tmp_path: Path) -> None:
    raiz = tmp_path / "d"
    ingest_csv_with_report(_csv(tmp_path / "a" / "ventas.csv"), data_root=raiz)
    nuevo = _csv(tmp_path / "b" / "ventas.csv", salto=3)

    salida = ingest_csv_with_report(nuevo, data_root=raiz, on_conflict="keep_both")

    assert salida.error is None
    datos = _json(salida.path)
    assert datos["status"] == "accepted"
    assert datos["deposit"]["status"] == "deposited"
    assert datos["deposit"]["would_overwrite"] is True
    assert datos["deposit"]["raw_path"] == f"raw/ventas-{_sha(nuevo)[:8]}.csv"
    assert (
        datos["published"]["parquet_path"]
        == f"processed/ventas-{_sha(nuevo)[:8]}.parquet"
    )
    assert len(list((raiz / "raw").iterdir())) == 2


# ---------------------------------------------------------------------------- fallas
def test_fila_invalida_deja_una_ingesta_fallida_con_su_fila(tmp_path: Path) -> None:
    origen = _csv(tmp_path / "e" / "v.csv", filas_extra=["FIL-0003,2026-01-01,5,abc"])

    salida = ingest_csv_with_report(origen, data_root=tmp_path / "d")

    assert isinstance(salida.error, SchemaBarrierError)
    datos = _json(salida.path)
    assert datos["status"] == "failed"
    assert datos["deposit"]["status"] == "deposited"
    assert datos["error"]["code"] == "ingest.schema_barrier"
    assert datos["error"]["source_exception"] == "SchemaBarrierError"
    assert datos["error"]["row_index"] is not None
    assert datos["error"]["column"] == "lead_time_days"
    assert datos["quality_log"][0]["row_index"] == datos["error"]["row_index"]
    assert datos["published"] is None


def test_sku_sin_demanda_positiva_es_una_falla_de_precondicion(tmp_path: Path) -> None:
    ceros = [f"FIL-0009,{date(2026, 1, 1) + timedelta(days=i)},0,7" for i in range(60)]
    origen = _csv(tmp_path / "e" / "v.csv", filas_extra=ceros)

    salida = ingest_csv_with_report(origen, data_root=tmp_path / "d")

    assert isinstance(salida.error, HandoffPreconditionError)
    datos = _json(salida.path)
    assert datos["status"] == "failed"
    assert datos["error"]["code"] == "handoff.precondition"


def test_cabeceras_rechazadas_dejan_el_informe_rechazado(tmp_path: Path) -> None:
    origen = _csv(tmp_path / "e" / "v.csv")
    rechazo = {
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

    salida = ingest_csv_with_report(
        origen,
        data_root=tmp_path / "d",
        provider=_Proveedor(rechazo),  # type: ignore[arg-type]
    )

    datos = _json(salida.path)
    assert datos["status"] == "rejected"
    assert datos["header_diagnostic"]["status"] == "rejected"
    assert (
        datos["header_diagnostic"]["entries"][0]["action"]
        == "Renombrar 'Date' a 'timestamp'"
    )
    assert datos["error"]["code"] == "ingest.header_rejected"
    assert datos["quality_log"][0]["column"] == "timestamp"


def test_ruta_con_proveedor_acepta_y_guarda_el_diagnostico(tmp_path: Path) -> None:
    origen = _csv(tmp_path / "e" / "v.csv")
    aceptado = {"status": "accepted", "diagnostic": []}

    salida = ingest_csv_with_report(
        origen,
        data_root=tmp_path / "d",
        provider=_Proveedor(aceptado),  # type: ignore[arg-type]
    )

    datos = _json(salida.path)
    assert datos["status"] == "accepted"
    assert datos["header_diagnostic"] == {"status": "accepted", "entries": []}


def test_sin_archivo_no_hay_informe_y_se_propaga(tmp_path: Path) -> None:
    raiz = tmp_path / "d"
    with pytest.raises(FileNotFoundError):
        ingest_csv_with_report(tmp_path / "no_existe.csv", data_root=raiz)
    assert not any((raiz / "ingestas").iterdir())


# ---------------------------------------------------------------------- codigos
@pytest.mark.parametrize(
    ("excepcion", "codigo"),
    [
        (
            RawConflictError(Path("raw/a.csv"), "a" * 64, "b" * 64),
            "ingest.raw_conflict",
        ),
        (RawWritePermissionError("raw/a.csv", "test"), "ingest.raw_write_blocked"),
        (SchemaBarrierError("x"), "ingest.schema_barrier"),
        (FileNotFoundError("x"), "ingest.file_not_found"),
        (
            ValueError("La extraccion pasiva solo admite archivos .csv"),
            "ingest.not_csv",
        ),
        (ValueError("otra cosa"), "pipeline.stage_failed"),
        (RuntimeError("inesperado"), "pipeline.stage_failed"),
    ],
)
def test_codigo_de_error_por_tipo(excepcion: BaseException, codigo: str) -> None:
    assert error_code(excepcion) == codigo


def test_ruta_del_informe_usa_el_hash(tmp_path: Path) -> None:
    sha = "ab" * 32
    assert (
        report_path(tmp_path / "d", sha) == tmp_path / "d" / "ingestas" / f"{sha}.json"
    )
