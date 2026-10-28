"""Verificacion local del motor antes de pedir revision (`make motor-local`).

Elige los niveles con las rutas que cambio la rama (`.github/motor-local.toml`),
los corre con `--workers 2` y 4 hebras de BLAS, escribe el recibo en
`local_data/recibos/<commit>.txt` y, con `--publicar`, lo publica como
comentario del PR con `gh pr comment`. CI lee ese comentario y fija el estado
`motor-local`. Ver docs/features/motor-local/README.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from recibo_motor import MAPA, Mapa, escribir_recibo, nombres

MOTOR = Path(__file__).resolve().parents[2]
SEMILLA = MOTOR / "tests" / "fixtures" / "semilla_inventario.csv"
VERIFICADOR = MOTOR / ".github" / "scripts" / "verificar_corrida.py"
CASO_PLATAFORMA = Path("tests") / "test_integracion_motor.py"
# Los argumentos M0 del trabajo `run` de CI (.github/workflows/ci.yml).
M0_HUMO = (
    "--m0-metodo mbb-directo --m0-n-series 1 --m0-minimo-filas 0"
    " --m0-columna sku_id=Item_ID --m0-columna timestamp=Date"
    " --m0-columna demand_qty=Avg_Usage_Per_Day"
    " --m0-columna lead_time_days=Restock_Lead_Time"
)
WORKERS = "2"
HEBRAS = "4"
CERROJO = Path("/tmp/pred-motor-local.lock")


@dataclass(frozen=True)
class Contexto:
    trabajo: Path
    plataforma: Path
    muestra22: Path


@dataclass
class Resultado:
    ok: bool
    salida: int
    duracion_s: float
    muestra: str
    detalles: dict[str, str] = field(default_factory=dict)

    def linea(self, nivel: str) -> str:
        campos = [
            "ok" if self.ok else "fallo",
            f"salida={self.salida}",
            f"duracion_s={round(self.duracion_s)}",
            f"muestra={self.muestra}",
            *(f"{clave}={valor}" for clave, valor in self.detalles.items()),
        ]
        return f"{nivel}: {' '.join(campos)}"


def _git(raiz: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(raiz), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _entorno(**extra: str) -> dict[str, str]:
    entorno = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        entorno[variable] = HEBRAS
    return entorno | extra


def _correr(comando: list[str], cwd: Path, log: Path, **entorno: str) -> int:
    with log.open("a", encoding="utf-8") as salida:
        salida.write(f"$ {shlex.join(comando)}\n")
        salida.flush()
        return subprocess.run(
            comando,
            cwd=cwd,
            stdout=salida,
            stderr=subprocess.STDOUT,
            env=_entorno(**entorno),
        ).returncode


def _pred_engine() -> str:
    comando = shutil.which("pred-engine", path=str(Path(sys.executable).parent))
    if comando is None:
        raise SystemExit("pred-engine no esta instalado en este entorno; use `uv run`")
    return comando


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _limpio(directorio: Path) -> Path:
    shutil.rmtree(directorio, ignore_errors=True)
    directorio.mkdir(parents=True)
    return directorio


def _motivo(paso: str, log: Path) -> str:
    """Primera causa legible del paso fallido, sin rutas de la maquina."""
    lineas = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    errores = [ln.split("::")[-1] for ln in lineas if ln.startswith("::error")]
    if errores:
        return f"{paso}: {errores[0]}"
    try:
        etapas = json.loads(lineas[-1])["stages"]
        fallidas = [
            f"{e['id']} {e['message']}" for e in etapas if e["state"] == "failed"
        ]
    except (IndexError, ValueError, KeyError, TypeError):
        fallidas = []
    return f"{paso}: {fallidas[0]}" if fallidas else paso


@dataclass(frozen=True)
class CorridaM3:
    salida: int
    duracion_s: float
    log: Path
    db: Path
    run_dir: Path
    motivo: str | None


def _corrida_m3(
    contexto: Contexto, nivel: str, entrada: list[str], reales: bool
) -> CorridaM3:
    """Base nueva de plataforma, `pred-engine run`, telemetria y verificador."""
    raiz = _limpio(contexto.trabajo / nivel)
    log, db, run_dir = raiz / "corrida.log", raiz / "pred.db", raiz / "runs" / nivel
    motor = _pred_engine()
    migrar = [sys.executable, "-m", "pred_platform.dal.migrate", "--db", str(db)]
    correr = [motor, "run", *entrada, "--data-root", str(raiz), "--db-path", str(db)]
    correr += ["--run-id", nivel, "--workers", WORKERS]
    verificar = [sys.executable, str(VERIFICADOR), str(run_dir), str(db)]
    verificar += ["--datos-reales"] * reales
    inicio = time.monotonic()

    def fin(salida: int, motivo: str | None = None) -> CorridaM3:
        return CorridaM3(salida, time.monotonic() - inicio, log, db, run_dir, motivo)

    plataforma = {"PYTHONPATH": str(contexto.plataforma / "src")}
    if codigo := _correr(migrar, MOTOR, log, **plataforma):
        return fin(codigo, _motivo("migracion de plataforma", log))
    salida = _correr(correr, MOTOR, log)
    if salida not in (0, 8):
        return fin(salida, _motivo("pred-engine run", log))
    for paso, comando in (
        ("pred-engine telemetry", [motor, "telemetry", str(run_dir)]),
        ("verificar_corrida.py", verificar),
    ):
        if _correr(comando, MOTOR, log):
            return fin(salida, _motivo(paso, log))
    return fin(salida)


def _veredictos(corrida: CorridaM3) -> tuple[dict[str, str], set[str]]:
    resumen = json.loads((corrida.run_dir / "corrida.json").read_text())["summary"]
    uri = f"{corrida.db.resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        filas = conn.execute(
            "SELECT sku, veredicto, candidato_campeon FROM m3_veredictos "
            "WHERE run_id = ?",
            (resumen["m3"]["run_id_m3"],),
        ).fetchall()
    # La huella permite comparar el recibo con otra corrida del mismo nivel.
    lineas = sorted(
        f"{sku}:{veredicto}:{candidato or ''}\n" for sku, veredicto, candidato in filas
    )
    conteo = Counter(veredicto for _, veredicto, _ in filas)
    return {
        "veredictos": ",".join(f"{v}:{n}" for v, n in sorted(conteo.items())),
        "huella": hashlib.sha256("".join(lineas).encode()).hexdigest()[:12],
    }, {sku for sku, _, _ in filas}


def _resultado_m3(corrida: CorridaM3, muestra: str) -> Resultado:
    if corrida.motivo:
        motivo = {"motivo": f'"{corrida.motivo}"'}
        return Resultado(False, corrida.salida, corrida.duracion_s, muestra, motivo)
    detalles, _ = _veredictos(corrida)
    return Resultado(True, corrida.salida, corrida.duracion_s, muestra, detalles)


def humo(contexto: Contexto) -> Resultado:
    """La corrida de CI: M0 `mbb-directo` sobre la semilla de 4 productos (8 SKU).

    Como CI, que no instala el extra foundation, corre sin Chronos-2.
    """
    entrada = ["--seed-csv", str(SEMILLA), "--no-foundation", *M0_HUMO.split()]
    corrida = _corrida_m3(contexto, "humo", entrada, reales=False)
    return _resultado_m3(corrida, f"humo-{_sha256(SEMILLA)[:12]}")


def muestra22(contexto: Contexto) -> Resultado:
    """Los 22 productos reales de `timing22.csv` con los ajustes por defecto."""
    muestra = f"m22-{_sha256(contexto.muestra22)[:12]}"
    entrada = ["--csv", str(contexto.muestra22), "--datos-reales"]
    corrida = _corrida_m3(contexto, "muestra22", entrada, reales=True)
    resultado = _resultado_m3(corrida, muestra)
    if resultado.ok:
        with contexto.muestra22.open(newline="", encoding="utf-8") as archivo:
            skus = {fila["sku_id"] for fila in csv.DictReader(archivo)}
        _, con_veredicto = _veredictos(corrida)
        if faltan := skus - con_veredicto:
            resultado.ok = False
            resultado.detalles["motivo"] = f"sin veredicto: {' '.join(sorted(faltan))}"
    return resultado


def plataforma(contexto: Contexto) -> Resultado:
    """El caso de integracion de pred-platform con este motor (0 omitidas)."""
    caso = (contexto.plataforma / CASO_PLATAFORMA).read_bytes()
    blob = hashlib.sha1(b"blob %d\0" % len(caso) + caso).hexdigest()
    raiz = _limpio(contexto.trabajo / "plataforma")
    log = raiz / "integracion.log"
    inicio = time.monotonic()
    salida = _correr(
        [
            "uv",
            "run",
            "--frozen",
            "--extra",
            "dev",
            "pytest",
            "-m",
            "integration",
            "--no-cov",
            "-ra",
        ],
        contexto.plataforma,
        log,
        PRED_ENGINE_CMD=shlex.join([_pred_engine()]),
    )
    duracion = time.monotonic() - inicio
    resumen = [
        ln
        for ln in log.read_text(encoding="utf-8").splitlines()
        if re.search(r"\d+ (passed|failed|error)", ln)
    ]
    final = resumen[-1].strip("= ") if resumen else _motivo("pytest", log)
    pasadas = re.search(r"(\d+) passed", final)
    ok = salida == 0 and pasadas is not None and "skipped" not in final
    return Resultado(
        ok, salida, duracion, f"plat6-{blob[:12]}", {"pruebas": f'"{final}"'}
    )


NIVELES: dict[str, Callable[[Contexto], Resultado]] = {
    "humo": humo,
    "plataforma": plataforma,
    "muestra22": muestra22,
}


def _equipo() -> str:
    nucleos = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count()
    )
    try:
        bytes_ = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        memoria = f"{bytes_ / 2**30:.0f} GiB"
    except (AttributeError, ValueError, OSError):
        memoria = "memoria desconocida"
    return f"{nucleos} nucleos, {memoria}, --workers {WORKERS}, hebras {HEBRAS}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base", default="origin/main", help="rama contra la que se compara"
    )
    parser.add_argument(
        "--nivel",
        action="append",
        default=[],
        choices=NIVELES,
        help="agrega un nivel a los que exigen las rutas (no quita ninguno)",
    )
    parser.add_argument(
        "--publicar", action="store_true", help="publica el recibo en el PR"
    )
    parser.add_argument("--plan", action="store_true", help="solo muestra los niveles")
    parser.add_argument(
        "--plataforma", type=Path, help="copia de pred-platform (../pred-platform)"
    )
    parser.add_argument(
        "--muestra22", type=Path, help="timing22.csv (local_data/muestras/timing22.csv)"
    )
    args = parser.parse_args(argv)

    raiz = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    if _git(raiz, "status", "--porcelain"):
        parser.error("hay cambios sin commit; el recibo debe nombrar un solo commit")
    mapa = Mapa.cargar(raiz / MAPA)
    if desconocidos := mapa.todos - NIVELES.keys():
        parser.error(f"{raiz / MAPA}: niveles desconocidos {nombres(desconocidos)}")
    rutas = _git(raiz, "diff", "--name-only", "--no-renames", f"{args.base}...HEAD")
    requeridos = mapa.requeridos(rutas.splitlines()) | set(args.nivel)
    commit = _git(raiz, "rev-parse", "HEAD")
    print(
        f"{commit[:7]} contra {args.base}: niveles {nombres(requeridos) or 'ninguno'}"
    )
    if args.plan:
        return 0
    if not requeridos:
        print("sin corrida: solo docs; CI marca motor-local sin recibo")
        return 0

    contexto = Contexto(
        trabajo=raiz / "local_data" / "motor-local",
        plataforma=(args.plataforma or raiz.parent / "pred-platform").resolve(),
        muestra22=(
            args.muestra22 or raiz / "local_data" / "muestras" / "timing22.csv"
        ).resolve(),
    )
    if not (contexto.plataforma / "src" / "pred_platform").is_dir():
        parser.error(
            f"{contexto.plataforma}: no es una copia de pred-platform (--plataforma)"
        )
    if (
        "plataforma" in requeridos
        and not (contexto.plataforma / CASO_PLATAFORMA).is_file()
    ):
        parser.error(
            f"{contexto.plataforma}: falta {CASO_PLATAFORMA}; actualice la copia"
        )
    if "muestra22" in requeridos and not contexto.muestra22.is_file():
        parser.error(
            f"{contexto.muestra22}: falta timing22.csv (--muestra22); "
            "ver docs/features/motor-local/README.md"
        )
    if args.publicar and shutil.which("gh") is None:
        parser.error("--publicar necesita gh; sin gh, pegue el recibo en el PR")

    try:
        os.mkdir(CERROJO)
    except FileExistsError:
        print(
            f"error: ya hay una corrida de motor-local en curso ({CERROJO}); "
            f"espere a que termine o elimine el cerrojo si el proceso anterior murió",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        lineas, corridos, ok = [], [], True
        for nivel, correr in NIVELES.items():
            if nivel not in requeridos:
                continue
            print(f"{nivel}: corriendo...", flush=True)
            resultado = correr(contexto)
            lineas.append(resultado.linea(nivel))
            corridos.append(nivel)
            print(lineas[-1], flush=True)
            if not resultado.ok:
                ok = False
                print(f"registros en {contexto.trabajo / nivel}", file=sys.stderr)
                break

        recibo = escribir_recibo(
            {
                "repo": mapa.repo,
                "commit": commit,
                "niveles": nombres(corridos),
                "resultado": "ok" if ok else "fallo",
                "motor": _git(MOTOR, "rev-parse", "HEAD"),
                "equipo": _equipo(),
                "fecha": datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ"),
            },
            lineas,
        )
        archivo = raiz / "local_data" / "recibos" / f"{commit}.txt"
        archivo.parent.mkdir(parents=True, exist_ok=True)
        archivo.write_text(recibo, encoding="utf-8")
        print(f"\n{recibo}\nrecibo en {archivo}")
        if args.publicar:
            cuerpo = f"Recibo de `make motor-local`:\n\n```text\n{recibo}```\n"
            subprocess.run(
                ["gh", "pr", "comment", "--body-file", "-"],
                cwd=raiz,
                input=cuerpo,
                text=True,
                check=True,
            )
        return 0 if ok else 1
    finally:
        shutil.rmtree(CERROJO, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
