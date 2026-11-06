"""Verificador de CI: versiones de la base central que acepta."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / ".github" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import verificar_corrida  # noqa: E402


def base(version: int) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(f"PRAGMA user_version = {version}")
    return conn


@pytest.mark.parametrize("version", [11, 12])
def test_acepta_la_version_del_motor_y_la_de_la_plataforma(version: int) -> None:
    assert verificar_corrida.version_base_valida(base(version))


@pytest.mark.parametrize("version", [4, 10, 13])
def test_rechaza_otras_versiones(version: int) -> None:
    assert not verificar_corrida.version_base_valida(base(version))
