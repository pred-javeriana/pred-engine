"""`make motor-local`: niveles de la rama y negativa con cambios sin commit."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
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


@pytest.fixture(autouse=True)
def cerrojo_aislado(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    lock = tmp_path / "pred-motor-local.lock"
    monkeypatch.setattr(motor_local, "CERROJO", lock, raising=False)
    return lock


def test_cerrojo_impide_segunda_corrida_concurrente(repo, monkeypatch, tmp_path):
    cerrojo = getattr(motor_local, "CERROJO", tmp_path / "pred-motor-local.lock")
    cerrojo.mkdir(parents=True, exist_ok=True)

    runner_llamado = False

    def fake_humo(ctx):
        nonlocal runner_llamado
        runner_llamado = True
        return motor_local.Resultado(True, 0, 1.0, "humo-123")

    monkeypatch.setitem(motor_local.NIVELES, "humo", fake_humo)

    plat_dir = tmp_path / "plat"
    (plat_dir / "src" / "pred_platform").mkdir(parents=True)

    with pytest.raises(SystemExit) as excinfo:
        motor_local.main(
            ["--base", "main", "--nivel", "humo", "--plataforma", str(plat_dir)]
        )

    assert excinfo.value.code != 0
    assert not runner_llamado


def test_cerrojo_se_libera_en_finally_ante_excepcion(repo, monkeypatch, tmp_path):
    cerrojo = getattr(motor_local, "CERROJO", tmp_path / "pred-motor-local.lock")

    def fake_falla(ctx):
        assert cerrojo.exists()
        raise RuntimeError("falla en runner")

    monkeypatch.setitem(motor_local.NIVELES, "humo", fake_falla)

    plat_dir = tmp_path / "plat"
    (plat_dir / "src" / "pred_platform").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="falla en runner"):
        motor_local.main(
            ["--base", "main", "--nivel", "humo", "--plataforma", str(plat_dir)]
        )

    assert not cerrojo.exists()


def test_muestra22_crea_directorio_padre_inexistente(tmp_path):
    filas = ["sku_id,timestamp,demand_qty"]
    for i in range(25):
        for dia in range(1, 11):
            filas.append(f"sku_{i:02d},2025-01-{dia:02d},10")
    origen = tmp_path / "panel_origen.csv"
    origen.write_text("\n".join(filas) + "\n", encoding="utf-8")

    destino = tmp_path / "subcarpeta" / "anidada" / "timing22.csv"
    assert not destino.parent.exists()

    resultado = subprocess.run(
        [sys.executable, str(SCRIPTS / "muestra22.py"), str(origen), str(destino)],
        capture_output=True,
        text=True,
    )
    assert resultado.returncode == 0
    assert destino.is_file()
    assert destino.stat().st_size > 0
    assert "m22-" in resultado.stdout


def test_cerrojo_independiente_de_tmpdir_en_subproceso(repo, monkeypatch, tmp_path):
    # Se utiliza una ruta fija aislada en /tmp mediante PRED_MOTOR_LOCAL_LOCK para
    # no interferir con otras corridas activas en la máquina ni depender de TMPDIR.
    token = uuid.uuid4().hex[:8]
    cerrojo_fijo = Path(f"/tmp/pred-motor-local-test-{token}.lock")
    monkeypatch.setattr(motor_local, "CERROJO", cerrojo_fijo)
    monkeypatch.setenv("PRED_MOTOR_LOCAL_LOCK", str(cerrojo_fijo))

    plat_dir = tmp_path / "plat"
    (plat_dir / "src" / "pred_platform").mkdir(parents=True)

    otro_tmp = tmp_path / "otro_tmpdir"
    otro_tmp.mkdir()

    resultado_subproceso: subprocess.CompletedProcess[str] | None = None

    def stub_primero(ctx):
        nonlocal resultado_subproceso
        assert cerrojo_fijo.exists()
        # Con el cerrojo activo en el proceso principal, lanzamos un segundo proceso
        # con un TMPDIR diferente para verificar que no toma un cerrojo separado.
        sub_codigo = f"""
import sys
from pathlib import Path

sys.path.insert(0, {str(SCRIPTS)!r})
import motor_local

stub_ejecutado = False

def stub_segundo(ctx):
    global stub_ejecutado
    stub_ejecutado = True
    return motor_local.Resultado(True, 0, 1.0, "humo-sub")

motor_local.NIVELES["humo"] = stub_segundo

try:
    ret = motor_local.main(
        ["--base", "main", "--nivel", "humo", "--plataforma", {str(plat_dir)!r}]
    )
    sys.exit(ret)
finally:
    if stub_ejecutado:
        sys.exit(99)
"""
        env = os.environ.copy()
        env["TMPDIR"] = str(otro_tmp)
        env["PRED_MOTOR_LOCAL_LOCK"] = str(cerrojo_fijo)
        resultado_subproceso = subprocess.run(
            [sys.executable, "-c", sub_codigo],
            cwd=str(repo),
            env=env,
            capture_output=True,
            text=True,
        )
        # El segundo proceso sale con error antes de entrar a su stub
        assert resultado_subproceso.returncode != 0
        assert resultado_subproceso.returncode != 99
        assert (
            "error: ya hay una corrida de motor-local en curso"
            in resultado_subproceso.stderr
        )
        assert str(cerrojo_fijo) in resultado_subproceso.stderr
        # El cerrojo del primer proceso sigue intacto mientras este ejecuta
        assert cerrojo_fijo.exists()
        return motor_local.Resultado(True, 0, 1.0, "humo-1")

    monkeypatch.setitem(motor_local.NIVELES, "humo", stub_primero)

    try:
        codigo = motor_local.main(
            ["--base", "main", "--nivel", "humo", "--plataforma", str(plat_dir)]
        )
        assert codigo == 0
        assert resultado_subproceso is not None
    finally:
        shutil.rmtree(cerrojo_fijo, ignore_errors=True)

    # Al terminar main(), el cerrojo debe haber sido liberado
    assert not cerrojo_fijo.exists()
