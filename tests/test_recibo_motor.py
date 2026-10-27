"""Control `motor-local`: niveles por ruta y estado a partir del recibo del PR."""

from __future__ import annotations

import sys
import urllib.error
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / ".github" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import recibo_motor  # noqa: E402

MAPA = recibo_motor.Mapa.cargar(SCRIPTS.parent / "motor-local.toml")
CABEZA = "c" * 40
ANTERIOR = "a" * 40


def recibo(commit: str = CABEZA, niveles: str = "muestra22", resultado: str = "ok"):
    texto = recibo_motor.escribir_recibo(
        {
            "repo": "pred-engine",
            "commit": commit,
            "niveles": niveles,
            "resultado": resultado,
        },
        [f"{n}: {resultado} salida=0" for n in niveles.split()],
    )
    return f"Recibo:\n\n```text\n{texto}```\n"


class FakeGitHub:
    def __init__(
        self,
        archivos: list[str],
        comentarios: list[tuple[str, str]],
        comparaciones: dict[str, tuple[str, list[str]]] | None = None,
        escritura: frozenset[str] = frozenset({"dev"}),
    ) -> None:
        self.archivos = archivos
        self.comentarios = comentarios
        self.comparaciones = comparaciones or {}
        self.escritura = escritura

    def get(self, ruta: str) -> Any:
        if ruta == "pulls/7":
            return {"head": {"sha": CABEZA}}
        if ruta.startswith("collaborators/"):
            usuario = ruta.split("/")[1]
            return {"permission": "write" if usuario in self.escritura else "read"}
        if ruta.startswith("compare/"):
            base = ruta.removeprefix("compare/").split("...")[0]
            if base not in self.comparaciones:
                raise urllib.error.HTTPError(ruta, 404, "Not Found", {}, None)  # type: ignore[arg-type]
            status, archivos = self.comparaciones[base]
            return {"status": status, "files": [{"filename": a} for a in archivos]}
        raise AssertionError(ruta)

    def paginar(self, ruta: str) -> Iterator[Any]:
        if ruta == "pulls/7/files":
            yield from ({"filename": a} for a in self.archivos)
        elif ruta == "issues/7/comments":
            for i, (autor, cuerpo) in enumerate(self.comentarios):
                yield {"user": {"login": autor}, "body": cuerpo, "html_url": f"c{i}"}
        else:
            raise AssertionError(ruta)


def estado(api: FakeGitHub) -> tuple[str, str]:
    resultado = recibo_motor.evaluar(api, 7, MAPA)
    assert resultado.sha == CABEZA
    return resultado.state, resultado.description


@pytest.mark.parametrize(
    ("rutas", "niveles"),
    [
        (["README.md", "docs/x.md", "tests/test_cli.py", ".github/ci.yml"], set()),
        (["src/pred_engine/forecasting/README.md"], set()),
        (["src/pred_engine/aumentacion/mbb.py", "Makefile"], {"humo"}),
        (["src/pred_engine/cli.py"], {"muestra22"}),
        (
            ["src/pred_engine/forecasting/persistencia_evidencia/esquema.py"],
            {"muestra22", "plataforma"},
        ),
        (["src/pred_engine/verify.py", "uv.lock"], {"humo", "muestra22"}),
        (["carpeta_nueva/modulo.py"], {"humo", "muestra22", "plataforma"}),
    ],
)
def test_niveles_por_ruta_primera_regla_y_union(rutas, niveles):
    assert MAPA.requeridos(rutas) == niveles


def test_pr_solo_docs_pasa_sin_recibo():
    assert estado(FakeGitHub(["docs/a.md"], [])) == (
        "success",
        "sin corrida: solo docs",
    )


def test_falta_recibo_nombra_niveles_y_commit():
    api = FakeGitHub(["src/pred_engine/cli.py"], [("dev", "sin recibo")])
    assert estado(api) == ("failure", "falta recibo muestra22 para ccccccc")


def test_recibo_ok_de_la_cabeza_pasa():
    api = FakeGitHub(["src/pred_engine/cli.py"], [("dev", recibo())])
    assert estado(api) == ("success", "recibo muestra22 de ccccccc")


def test_recibo_con_fallo_o_nivel_menor_no_pasa():
    fallo = FakeGitHub(["src/pred_engine/cli.py"], [("dev", recibo(resultado="fallo"))])
    assert estado(fallo) == ("failure", "recibo de ccccccc con resultado fallo")
    menor = FakeGitHub(["src/pred_engine/cli.py"], [("dev", recibo(niveles="humo"))])
    assert estado(menor) == ("failure", "recibo de ccccccc sin muestra22")


def test_decide_el_recibo_mas_reciente():
    api = FakeGitHub(
        ["src/pred_engine/cli.py"],
        [("dev", recibo(resultado="fallo")), ("dev", recibo(niveles="humo muestra22"))],
    )
    assert estado(api) == ("success", "recibo humo muestra22 de ccccccc")


def test_ignora_recibos_sin_permiso_de_escritura_o_de_otro_repo():
    otro_repo = recibo().replace("repo: pred-engine", "repo: pred-platform")
    api = FakeGitHub(
        ["src/pred_engine/cli.py"], [("visita", recibo()), ("dev", otro_repo)]
    )
    assert estado(api) == ("failure", "falta recibo muestra22 para ccccccc")


def test_recibo_anterior_vale_si_despues_solo_cambio_docs():
    api = FakeGitHub(
        ["src/pred_engine/cli.py", "README.md"],
        [("dev", recibo(commit=ANTERIOR))],
        {ANTERIOR: ("ahead", ["README.md", "docs/x.md"])},
    )
    assert estado(api) == (
        "success",
        "recibo muestra22 de aaaaaaa; cambios posteriores solo docs",
    )


@pytest.mark.parametrize(
    "comparacion",
    [
        ("ahead", ["src/pred_engine/cli.py"]),  # cambio de codigo despues del recibo
        ("diverged", ["README.md"]),  # rebase: el recibo ya no es ancestro
        ("ahead", [f"docs/{i}.md" for i in range(300)]),  # comparacion truncada
    ],
)
def test_recibo_anterior_queda_vencido(comparacion):
    api = FakeGitHub(
        ["src/pred_engine/cli.py"],
        [("dev", recibo(commit=ANTERIOR))],
        {ANTERIOR: comparacion},
    )
    assert estado(api) == ("failure", "falta recibo muestra22 para ccccccc")


def test_commit_desconocido_no_cuenta():
    api = FakeGitHub(["src/pred_engine/cli.py"], [("dev", recibo(commit=ANTERIOR))])
    assert estado(api)[0] == "failure"


def test_recibo_ida_y_vuelta():
    campos = {"repo": "pred-engine", "commit": CABEZA, "niveles": "humo"}
    texto = recibo_motor.escribir_recibo(campos, ["humo: ok salida=0"])
    comentario = f"```text\n{texto}```\notro texto\n```\n{texto}```"
    leidos = recibo_motor.leer_recibos(comentario.replace("\n", "\r\n"))
    assert leidos == [campos | {"humo": "ok salida=0"}] * 2
