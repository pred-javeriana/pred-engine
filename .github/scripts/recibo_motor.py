"""Recibo de la verificacion local del motor y control `motor-local` de CI.

Este modulo es el dueno del contrato que comparten `make motor-local` y CI: el
mapa de niveles por ruta (`.github/motor-local.toml`) y el formato del recibo.
Como comando, fija el estado `motor-local` del commit cabeza de un PR a partir
del recibo publicado como comentario; no corre el motor ni codigo del PR.

Uso: `python .github/scripts/recibo_motor.py [--pr N] [--publicar]`. Sin
`--publicar` solo imprime el estado. Solo usa la biblioteca estandar.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

FORMATO = "recibo-motor-local/1"
CONTEXTO = "motor-local"
MAPA = Path(".github/motor-local.toml")
_BLOQUE = re.compile(r"```[a-z]*\n(" + re.escape(FORMATO) + r"\n.*?)```", re.DOTALL)
# El API de comparacion devuelve como maximo 300 archivos; con mas no se sabe
# si los cambios posteriores al recibo son solo docs.
_MAX_ARCHIVOS_COMPARACION = 300


def _patron(glob: str) -> re.Pattern[str]:
    partes = re.split(r"(\*\*/|\*\*|\*|\?)", glob)
    traduccion = {"**/": "(?:.*/)?", "**": ".*", "*": "[^/]*", "?": "[^/]"}
    return re.compile("".join(traduccion.get(p, re.escape(p)) for p in partes))


@dataclass(frozen=True)
class Mapa:
    """Reglas ordenadas: cada ruta toma los niveles de la primera que coincide."""

    repo: str
    reglas: tuple[tuple[tuple[re.Pattern[str], ...], frozenset[str]], ...]

    @classmethod
    def cargar(cls, ruta: Path = MAPA) -> Mapa:
        datos = tomllib.loads(ruta.read_text(encoding="utf-8"))
        return cls(
            repo=datos["repo"],
            reglas=tuple(
                (tuple(_patron(r) for r in regla["rutas"]), frozenset(regla["niveles"]))
                for regla in datos["reglas"]
            ),
        )

    @property
    def todos(self) -> frozenset[str]:
        return frozenset().union(*(niveles for _, niveles in self.reglas))

    def niveles(self, ruta: str) -> frozenset[str]:
        for patrones, niveles in self.reglas:
            if any(p.fullmatch(ruta) for p in patrones):
                return niveles
        return self.todos

    def requeridos(self, rutas: Iterable[str]) -> frozenset[str]:
        return frozenset().union(*(self.niveles(r) for r in rutas))


def escribir_recibo(campos: dict[str, str], lineas_nivel: Iterable[str]) -> str:
    """Texto del recibo: cabecera `clave: valor` y una linea por nivel corrido."""
    cabecera = [FORMATO, *(f"{clave}: {valor}" for clave, valor in campos.items())]
    return "\n".join([*cabecera, "", *lineas_nivel]) + "\n"


def leer_recibos(texto: str) -> list[dict[str, str]]:
    """Recibos de un comentario, en orden; cada uno como `clave: valor`."""
    recibos = []
    for bloque in _BLOQUE.findall(texto.replace("\r\n", "\n")):
        campos = {}
        for linea in bloque.splitlines()[1:]:
            clave, separador, valor = linea.partition(": ")
            if separador:
                campos[clave.strip()] = valor.strip()
        recibos.append(campos)
    return recibos


def nombres(niveles: Iterable[str]) -> str:
    return " ".join(sorted(niveles))


class Api(Protocol):
    def get(self, ruta: str) -> Any: ...

    def paginar(self, ruta: str) -> Iterator[Any]: ...


class GitHub:
    """Cliente minimo del API REST de GitHub para un repositorio."""

    def __init__(self, repo: str, token: str | None) -> None:
        self.base = f"https://api.github.com/repos/{repo}/"
        self.cabeceras = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            self.cabeceras["Authorization"] = f"Bearer {token}"

    def _pedir(self, ruta: str, cuerpo: dict[str, str] | None = None) -> Any:
        datos = None if cuerpo is None else json.dumps(cuerpo).encode()
        pedido = urllib.request.Request(
            self.base + ruta, data=datos, headers=self.cabeceras
        )
        with urllib.request.urlopen(pedido, timeout=30) as respuesta:
            return json.load(respuesta)

    def get(self, ruta: str) -> Any:
        return self._pedir(ruta)

    def paginar(self, ruta: str) -> Iterator[Any]:
        separador = "&" if "?" in ruta else "?"
        pagina = 1
        while True:
            lote = self._pedir(f"{ruta}{separador}per_page=100&page={pagina}")
            yield from lote
            if len(lote) < 100:
                return
            pagina += 1

    def post(self, ruta: str, cuerpo: dict[str, str]) -> Any:
        return self._pedir(ruta, cuerpo)


@dataclass(frozen=True)
class Estado:
    sha: str
    state: str
    description: str
    target_url: str | None = None


def _rutas(archivos: Iterable[dict[str, Any]]) -> set[str]:
    rutas = set()
    for archivo in archivos:
        rutas.add(archivo["filename"])
        if archivo.get("previous_filename"):
            rutas.add(archivo["previous_filename"])
    return rutas


def evaluar(api: Api, numero: int, mapa: Mapa) -> Estado:
    """Estado `motor-local` del commit cabeza del PR `numero`.

    Decide el recibo mas reciente de un autor con escritura que nombra la
    cabeza, o un ancestro cuyos cambios posteriores no exigen ningun nivel.
    """
    cabeza = api.get(f"pulls/{numero}")["head"]["sha"]
    corto = cabeza[:7]
    requeridos = mapa.requeridos(_rutas(api.paginar(f"pulls/{numero}/files")))
    if not requeridos:
        return Estado(cabeza, "success", "sin corrida: solo docs")

    permisos: dict[str, bool] = {}

    def con_escritura(usuario: str) -> bool:
        if usuario not in permisos:
            permiso = api.get(f"collaborators/{usuario}/permission")["permission"]
            permisos[usuario] = permiso in ("admin", "write")
        return permisos[usuario]

    def solo_docs_desde(commit: str) -> bool:
        try:
            comparacion = api.get(f"compare/{commit}...{cabeza}")
        except urllib.error.HTTPError:
            return False
        archivos = comparacion.get("files", [])
        return (
            comparacion["status"] == "ahead"
            and len(archivos) < _MAX_ARCHIVOS_COMPARACION
            and not mapa.requeridos(_rutas(archivos))
        )

    comentarios = list(api.paginar(f"issues/{numero}/comments"))
    for comentario in reversed(comentarios):
        for recibo in reversed(leer_recibos(comentario["body"] or "")):
            commit = recibo.get("commit", "")
            if recibo.get("repo") != mapa.repo or not re.fullmatch(
                r"[0-9a-f]{40}", commit
            ):
                continue
            if not con_escritura(comentario["user"]["login"]):
                continue
            if commit == cabeza:
                nota = ""
            elif solo_docs_desde(commit):
                nota = "; cambios posteriores solo docs"
            else:
                continue
            url = comentario["html_url"]
            if recibo.get("resultado") != "ok":
                return Estado(
                    cabeza,
                    "failure",
                    f"recibo de {commit[:7]} con resultado "
                    f"{recibo.get('resultado', '?')}",
                    url,
                )
            corridos = frozenset(recibo.get("niveles", "").split())
            if faltan := requeridos - corridos:
                return Estado(
                    cabeza,
                    "failure",
                    f"recibo de {commit[:7]} sin {nombres(faltan)}",
                    url,
                )
            return Estado(
                cabeza,
                "success",
                f"recibo {nombres(corridos)} de {commit[:7]}{nota}",
                url,
            )
    return Estado(cabeza, "failure", f"falta recibo {nombres(requeridos)} para {corto}")


def _numero_del_evento() -> int:
    evento = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    return int((evento.get("pull_request") or evento["issue"])["number"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--pr", type=int, help="por defecto, el PR del evento de CI")
    parser.add_argument("--mapa", type=Path, default=MAPA)
    parser.add_argument(
        "--publicar", action="store_true", help="fija el estado en GitHub"
    )
    args = parser.parse_args(argv)
    if not args.repo:
        parser.error("falta --repo (o GITHUB_REPOSITORY)")
    api = GitHub(args.repo, os.environ.get("GITHUB_TOKEN"))
    estado = evaluar(api, args.pr or _numero_del_evento(), Mapa.cargar(args.mapa))
    print(f"{CONTEXTO} {estado.sha[:7]}: {estado.state} - {estado.description}")
    if args.publicar:
        cuerpo = {
            "state": estado.state,
            "context": CONTEXTO,
            "description": estado.description[:140],
        }
        if estado.target_url:
            cuerpo["target_url"] = estado.target_url
        api.post(f"statuses/{estado.sha}", cuerpo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
