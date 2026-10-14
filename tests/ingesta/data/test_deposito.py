"""Deposito en raw/ sin pisar archivos (G5, ADR-01-002)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pred_engine.ingesta.data import (
    RawConflictError,
    deposit_raw,
    ensure_data_layout,
    versioned_name,
)
from pred_engine.ingesta.pipeline import deposit_raw_csv


def _escribir(ruta: Path, texto: str) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")
    return ruta


def _sha(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _archivos_raw(raiz: Path) -> list[str]:
    return sorted(p.name for p in (raiz / "raw").iterdir())


def test_archivo_nuevo_se_deposita_con_su_hash(tmp_path: Path) -> None:
    origen = _escribir(tmp_path / "entrada" / "ventas.csv", "a,b\n1,2\n")
    raiz = tmp_path / "data"

    resultado = deposit_raw(origen, data_root=raiz)

    assert resultado.status == "deposited"
    assert resultado.would_overwrite is False
    assert resultado.sha256 == _sha("a,b\n1,2\n")
    assert resultado.path == (raiz / "raw" / "ventas.csv").resolve()
    assert resultado.path.read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_mismo_contenido_y_nombre_se_reutiliza_sin_reescribir(tmp_path: Path) -> None:
    origen = _escribir(tmp_path / "entrada" / "ventas.csv", "a,b\n1,2\n")
    raiz = tmp_path / "data"
    primero = deposit_raw(origen, data_root=raiz)
    antes = primero.path.stat().st_mtime_ns

    segundo = deposit_raw(origen, data_root=raiz)

    assert segundo.status == "already_present_identical"
    assert segundo.would_overwrite is False
    assert segundo.path == primero.path
    assert segundo.path.stat().st_mtime_ns == antes


def test_mismo_contenido_desde_otra_ruta_es_identico(tmp_path: Path) -> None:
    raiz = tmp_path / "data"
    deposit_raw(_escribir(tmp_path / "uno" / "ventas.csv", "x\n"), data_root=raiz)

    segundo = deposit_raw(
        _escribir(tmp_path / "dos" / "ventas.csv", "x\n"), data_root=raiz
    )

    assert segundo.status == "already_present_identical"
    assert _archivos_raw(raiz) == ["ventas.csv"]


def test_archivo_ya_dentro_de_raw_no_se_copia(tmp_path: Path) -> None:
    raiz = tmp_path / "data"
    layout = ensure_data_layout(raiz)
    dentro = _escribir(layout.raw / "ventas.csv", "x\n")

    resultado = deposit_raw(dentro, data_root=raiz)

    assert resultado.status == "already_present_identical"
    assert _archivos_raw(raiz) == ["ventas.csv"]


def test_conflicto_se_rechaza_sin_escribir_nada(tmp_path: Path) -> None:
    raiz = tmp_path / "data"
    original = deposit_raw(
        _escribir(tmp_path / "a" / "ventas.csv", "viejo\n"), data_root=raiz
    )
    nuevo = _escribir(tmp_path / "b" / "ventas.csv", "nuevo\n")

    with pytest.raises(RawConflictError) as capturado:
        deposit_raw(nuevo, data_root=raiz)

    error = capturado.value
    assert isinstance(error, FileExistsError)
    assert error.existing == original.path
    assert error.existing_sha256 == _sha("viejo\n")
    assert error.incoming_sha256 == _sha("nuevo\n")
    assert original.path.read_text(encoding="utf-8") == "viejo\n"
    assert _archivos_raw(raiz) == ["ventas.csv"]


def test_conservar_ambos_deposita_con_nombre_versionado(tmp_path: Path) -> None:
    raiz = tmp_path / "data"
    original = deposit_raw(
        _escribir(tmp_path / "a" / "ventas.csv", "viejo\n"), data_root=raiz
    )
    nuevo = _escribir(tmp_path / "b" / "ventas.csv", "nuevo\n")

    resultado = deposit_raw(nuevo, data_root=raiz, on_conflict="keep_both")

    esperado = versioned_name("ventas.csv", _sha("nuevo\n"))
    assert resultado.status == "deposited"
    assert resultado.would_overwrite is True
    assert resultado.path.name == esperado
    assert resultado.path.read_text(encoding="utf-8") == "nuevo\n"
    assert original.path.read_text(encoding="utf-8") == "viejo\n"
    assert _archivos_raw(raiz) == sorted(["ventas.csv", esperado])


def test_conservar_ambos_dos_veces_es_idempotente(tmp_path: Path) -> None:
    raiz = tmp_path / "data"
    deposit_raw(_escribir(tmp_path / "a" / "ventas.csv", "viejo\n"), data_root=raiz)
    nuevo = _escribir(tmp_path / "b" / "ventas.csv", "nuevo\n")
    primero = deposit_raw(nuevo, data_root=raiz, on_conflict="keep_both")

    segundo = deposit_raw(nuevo, data_root=raiz, on_conflict="keep_both")

    assert segundo.status == "already_present_identical"
    assert segundo.path == primero.path
    assert len(_archivos_raw(raiz)) == 2


def test_no_quedan_temporales_tras_depositar_ni_tras_un_conflicto(
    tmp_path: Path,
) -> None:
    raiz = tmp_path / "data"
    deposit_raw(_escribir(tmp_path / "a" / "ventas.csv", "viejo\n"), data_root=raiz)
    with pytest.raises(RawConflictError):
        deposit_raw(_escribir(tmp_path / "b" / "ventas.csv", "nuevo\n"), data_root=raiz)
    assert not [n for n in _archivos_raw(raiz) if n.endswith(".part")]


def test_origen_inexistente(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        deposit_raw(tmp_path / "no_existe.csv", data_root=tmp_path / "data")


def test_nombre_versionado_conserva_extension_y_usa_ocho_caracteres() -> None:
    sha = "0123456789abcdef" * 4
    assert versioned_name("ventas.csv", sha) == "ventas-01234567.csv"
    assert versioned_name("sin_extension", sha) == "sin_extension-01234567"


def test_deposit_raw_csv_conserva_su_contrato_y_no_sobrescribe(tmp_path: Path) -> None:
    raiz = tmp_path / "data"
    ruta = deposit_raw_csv(
        _escribir(tmp_path / "a" / "ventas.csv", "viejo\n"), data_root=raiz
    )
    assert ruta == (raiz / "raw" / "ventas.csv").resolve()

    with pytest.raises(RawConflictError):
        deposit_raw_csv(
            _escribir(tmp_path / "b" / "ventas.csv", "nuevo\n"), data_root=raiz
        )
    assert ruta.read_text(encoding="utf-8") == "viejo\n"
