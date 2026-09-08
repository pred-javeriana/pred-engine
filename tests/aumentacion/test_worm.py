"""0.4-B1 - Pruebas de la guarda de escritura WORM."""

from __future__ import annotations

from pathlib import Path

import pytest

from pred_engine.aumentacion.errores import WormOverwriteError
from pred_engine.aumentacion.worm import (
    escribir_una_sola_vez,
    resolver_ruta_artefacto,
)


def _escritor(contenido: str):
    def escribir(destino: Path) -> None:
        destino.write_text(contenido, encoding="utf-8")

    return escribir


def test_primera_escritura_deposita_el_artefacto(tmp_path: Path) -> None:
    destino = escribir_una_sola_vez(
        "panel.csv", _escritor("ok"), data_root=tmp_path / "data"
    )
    assert destino == (tmp_path / "data" / "raw" / "panel.csv")
    assert destino.read_text(encoding="utf-8") == "ok"


def test_segundo_intento_es_rechazado(tmp_path: Path) -> None:
    escribir_una_sola_vez("panel.csv", _escritor("v1"), data_root=tmp_path / "data")
    with pytest.raises(WormOverwriteError):
        escribir_una_sola_vez("panel.csv", _escritor("v2"), data_root=tmp_path / "data")
    ruta = tmp_path / "data" / "raw" / "panel.csv"
    assert ruta.read_text(encoding="utf-8") == "v1"


def test_ruta_absoluta_es_rechazada(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absoluta"):
        resolver_ruta_artefacto("/etc/passwd", data_root=tmp_path / "data")


def test_nombre_con_directorio_es_rechazado(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="directorios"):
        resolver_ruta_artefacto("../fuera.csv", data_root=tmp_path / "data")


def test_data_root_desde_variable_de_entorno(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRED_DATA_ROOT", str(tmp_path / "env-data"))
    destino = resolver_ruta_artefacto("panel.csv")
    assert destino == (tmp_path / "env-data" / "raw" / "panel.csv")


def test_escritor_que_no_materializa_archivo_es_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        escribir_una_sola_vez(
            "panel.csv", lambda _destino: None, data_root=tmp_path / "data"
        )
