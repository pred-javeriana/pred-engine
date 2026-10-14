"""Ruta canonica (sin LLM): deposita en raw/, devuelve el hash y no pisa archivos."""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from pred_engine.ingesta.data import RawConflictError
from pred_engine.ingesta.pipeline import (
    ingest_canonical_csv,
    run_classify_csv,
    run_ingest,
)


def _csv(ruta: Path, *, salto: int = 0, dias: int = 60) -> Path:
    """Dos SKU diarios; ``salto`` cambia el contenido sin cambiar el nombre."""
    inicio = date(2026, 1, 1)
    filas = ["sku_id,timestamp,demand_qty,lead_time_days"]
    for i in range(dias):
        dia = (inicio + timedelta(days=i)).isoformat()
        filas.append(f"FIL-0001,{dia},{5 + i % 7 + salto},7")
        filas.append(f"FIL-0002,{dia},{4 if i % 3 == 0 else 0},7")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(filas) + "\n", encoding="utf-8")
    return ruta


def _sha(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


class _Proveedor:
    def complete(self, prompt: str, *, temperature: float, timeout: float) -> str:
        return json.dumps({"status": "accepted", "diagnostic": []})


def test_deposita_y_devuelve_el_hash_de_la_fuente(tmp_path: Path) -> None:
    origen = _csv(tmp_path / "entrada" / "ventas.csv")
    raiz = tmp_path / "data"

    resultado = ingest_canonical_csv(origen, data_root=raiz)

    assert resultado.source_sha256 == _sha(origen)
    assert resultado.deposit.status == "deposited"
    assert resultado.deposit.path == (raiz / "raw" / "ventas.csv").resolve()
    assert resultado.deposit.path.read_bytes() == origen.read_bytes()
    assert resultado.row_count == 120
    assert resultado.parquet_path == raiz / "processed" / "ventas.parquet"
    assert resultado.parquet_path.is_file()
    assert {m.sku_id for m in resultado.topology.metrics} == {"FIL-0001", "FIL-0002"}


def test_repetir_la_misma_ingesta_es_idempotente(tmp_path: Path) -> None:
    origen = _csv(tmp_path / "entrada" / "ventas.csv")
    raiz = tmp_path / "data"
    primero = ingest_canonical_csv(origen, data_root=raiz)

    segundo = ingest_canonical_csv(origen, data_root=raiz)

    assert segundo.deposit.status == "already_present_identical"
    assert segundo.source_sha256 == primero.source_sha256
    assert sorted(p.name for p in (raiz / "raw").iterdir()) == ["ventas.csv"]


def test_nombre_repetido_con_otro_contenido_se_rechaza_y_no_publica(
    tmp_path: Path,
) -> None:
    raiz = tmp_path / "data"
    ingest_canonical_csv(_csv(tmp_path / "a" / "ventas.csv"), data_root=raiz)
    parquet = raiz / "processed" / "ventas.parquet"
    antes = parquet.read_bytes()
    otro = _csv(tmp_path / "b" / "ventas.csv", salto=3)

    with pytest.raises(RawConflictError):
        ingest_canonical_csv(otro, data_root=raiz)

    assert parquet.read_bytes() == antes
    assert (raiz / "raw" / "ventas.csv").read_bytes() != otro.read_bytes()


def test_conservar_ambos_deja_dos_fuentes_y_dos_parquets(tmp_path: Path) -> None:
    raiz = tmp_path / "data"
    primero = ingest_canonical_csv(_csv(tmp_path / "a" / "ventas.csv"), data_root=raiz)
    otro = _csv(tmp_path / "b" / "ventas.csv", salto=3)

    segundo = ingest_canonical_csv(otro, data_root=raiz, on_conflict="keep_both")

    assert segundo.deposit.would_overwrite is True
    assert segundo.deposit.path != primero.deposit.path
    assert segundo.parquet_path != primero.parquet_path
    assert primero.parquet_path.is_file() and segundo.parquet_path.is_file()
    assert segundo.source_sha256 == _sha(otro)
    assert (
        primero.deposit.path.read_bytes()
        == (tmp_path / "a" / "ventas.csv").read_bytes()
    )


def test_run_classify_csv_mantiene_su_firma_y_ahora_deposita(tmp_path: Path) -> None:
    origen = _csv(tmp_path / "entrada" / "ventas.csv")
    raiz = tmp_path / "data"

    topologia, parquet = run_classify_csv(origen, data_root=raiz)

    assert len(topologia.metrics) == 2
    assert parquet == raiz / "processed" / "ventas.parquet"
    assert (raiz / "raw" / "ventas.csv").is_file()


def test_run_ingest_expone_el_deposito_y_el_mismo_hash(tmp_path: Path) -> None:
    origen = _csv(tmp_path / "entrada" / "ventas.csv")

    resultado = run_ingest(origen, _Proveedor(), data_root=tmp_path / "data")  # type: ignore[arg-type]

    assert resultado.deposit is not None
    assert resultado.deposit.sha256 == resultado.source.sha256 == _sha(origen)


def test_run_ingest_rechaza_un_conflicto_y_con_confirmacion_conserva_ambos(
    tmp_path: Path,
) -> None:
    raiz = tmp_path / "data"
    run_ingest(_csv(tmp_path / "a" / "ventas.csv"), _Proveedor(), data_root=raiz)  # type: ignore[arg-type]
    otro = _csv(tmp_path / "b" / "ventas.csv", salto=3)

    with pytest.raises(RawConflictError):
        run_ingest(otro, _Proveedor(), data_root=raiz)  # type: ignore[arg-type]

    resultado = run_ingest(
        otro,
        _Proveedor(),  # type: ignore[arg-type]
        data_root=raiz,
        on_conflict="keep_both",
    )
    assert resultado.deposit is not None and resultado.deposit.would_overwrite is True
    assert len(list((raiz / "raw").iterdir())) == 2
