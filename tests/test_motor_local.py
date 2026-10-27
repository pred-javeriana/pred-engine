"""`make motor-local`: niveles de la rama y negativa con cambios sin commit."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / ".github" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import motor_local  # noqa: E402


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / ".github").mkdir()
    shutil.copy(SCRIPTS.parent / "motor-local.toml", tmp_path / ".github")
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    git(tmp_path, "checkout", "-qb", "rama")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def commit(repo: Path, ruta: str) -> None:
    archivo = repo / ruta
    archivo.parent.mkdir(parents=True, exist_ok=True)
    archivo.write_text("x\n")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", ruta)


def test_rama_solo_docs_no_corre_nada(repo, capsys):
    commit(repo, "docs/nota.md")
    assert motor_local.main(["--base", "main"]) == 0
    salida = capsys.readouterr().out
    assert "niveles ninguno" in salida
    assert "sin corrida: solo docs" in salida
    assert not (repo / "local_data").exists()


def test_niveles_de_las_rutas_mas_los_pedidos(repo, capsys):
    commit(repo, "src/pred_engine/aumentacion/mbb.py")
    argv = ["--base", "main", "--plan", "--nivel", "plataforma"]
    assert motor_local.main(argv) == 0
    assert "niveles humo plataforma" in capsys.readouterr().out


def test_rechaza_cambios_sin_commit(repo):
    commit(repo, "src/pred_engine/cli.py")
    (repo / "src/pred_engine/cli.py").write_text("cambio\n")
    with pytest.raises(SystemExit):
        motor_local.main(["--base", "main", "--plan"])
