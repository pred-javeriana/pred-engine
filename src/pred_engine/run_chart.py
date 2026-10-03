"""Grafico SVG de una corrida terminada: recursos en el tiempo y traza de modelos.

Lee los artefactos que deja ``pred-engine run`` en el directorio de la corrida
(``unidades.jsonl``, ``recursos.jsonl`` y ``corrida.json``) y escribe un SVG
autocontenido, sin dependencias de graficacion. De arriba abajo, sobre el mismo
eje de tiempo:

1. Etapas L1-L3.
2. CPU en nucleos: procesos de la corrida, maquina entera y capacidad.
3. Memoria: RSS de los procesos de la corrida, memoria en uso de la maquina y
   memoria total.
4. Memoria residente de cada proceso (principal y procesos del pool).
5. Una fila por proceso con cada unidad SKU x familia, coloreada por modelo;
   pasar el puntero sobre una barra muestra SKU, etapa, duracion y CPU.

Sin ``recursos.jsonl`` (muestreo desactivado) solo se dibujan las etapas y las
unidades.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

CHART_NAME = "telemetria.svg"

# Paleta categorica en orden fijo: el color sigue al modelo, no a su rango.
_MODELOS = ("sarima", "lightgbm", "mlp", "chronos2")

_ANCHO = 1200
_IZQ, _DER = 84, 120
_PANEL = 132
_SEPARACION = 46
_CARRIL, _HUECO = 10, 3


def _epoch(instante: str) -> float:
    return datetime.fromisoformat(instante).timestamp()


def _leer_jsonl(ruta: Path) -> list[dict[str, Any]]:
    if not ruta.exists():
        return []
    with ruta.open(encoding="utf-8") as archivo:
        return [json.loads(linea) for linea in archivo if linea.strip()]


def _paso_tiempo(duracion: float) -> float:
    for paso in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200):
        if duracion / paso <= 10:
            return paso
    return 14400


def _paso_valor(maximo: float) -> float:
    """Paso redondo (1, 2, 2.5 o 5 por potencia de 10) con a lo sumo 7 marcas."""
    exponente = math.floor(math.log10(maximo)) - 1
    while True:
        for factor in (1, 2, 2.5, 5):
            paso = factor * 10.0**exponente
            if maximo / paso <= 7:
                return paso
        exponente += 1


def _reloj(segundos: float) -> str:
    minutos, resto = divmod(int(round(segundos)), 60)
    return f"{minutos}:{resto:02d}"


def _duracion(segundos: float) -> str:
    minutos, resto = divmod(int(round(segundos)), 60)
    return f"{minutos} min {resto} s" if minutos else f"{segundos:.1f} s"


def _numero(valor: float) -> str:
    return f"{valor:g}" if valor >= 1 or valor == 0 else f"{valor:.2g}"


class _Lienzo:
    def __init__(self, t0: float, t1: float) -> None:
        self.t0 = t0
        self.duracion = max(t1 - t0, 1e-6)
        self.ancho = _ANCHO - _IZQ - _DER
        self.partes: list[str] = []

    def x(self, epoch: float) -> float:
        return round(_IZQ + (epoch - self.t0) / self.duracion * self.ancho, 2)

    def add(self, fragmento: str) -> None:
        self.partes.append(fragmento)

    def texto(self, x: float, y: float, contenido: str, clase: str, **attrs: Any):
        extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        self.add(
            f'<text x="{x}" y="{y}" class="{clase}"{extra}>{escape(contenido)}</text>'
        )


def _panel(
    lienzo: _Lienzo,
    y: float,
    titulo: str,
    unidad: str,
    maximo: float,
    series: Sequence[tuple[str, Sequence[tuple[float, float]]]],
    capacidad: tuple[float, str] | None = None,
    leyenda: Sequence[tuple[str, str]] = (),
) -> None:
    """Panel de lineas con un solo eje; ``series`` son (clase, puntos)."""
    alto = _PANEL
    maximo = max(maximo, 1e-6)
    paso = _paso_valor(maximo)
    tope = paso * (int(maximo / paso - 1e-9) + 1)

    def py(valor: float) -> float:
        return round(y + alto - valor / tope * alto, 2)

    lienzo.texto(_IZQ, y - 14, titulo, "titulo-panel")
    x_leyenda = _IZQ + 8 * len(titulo) + 24
    for clase, rotulo in leyenda:
        lienzo.add(
            f'<line x1="{x_leyenda}" x2="{x_leyenda + 16}" y1="{y - 18}" '
            f'y2="{y - 18}" class="{clase}"/>'
        )
        lienzo.texto(x_leyenda + 22, y - 14, rotulo, "eje")
        x_leyenda += 30 + 7 * len(rotulo)
    marca = 0.0
    while marca <= tope + 1e-9:
        lienzo.add(
            f'<line x1="{_IZQ}" x2="{_IZQ + lienzo.ancho}" y1="{py(marca)}" '
            f'y2="{py(marca)}" class="rejilla"/>'
        )
        lienzo.texto(
            _IZQ - 8, py(marca) + 4, _numero(round(marca, 6)), "eje", text_anchor="end"
        )
        marca += paso
    lienzo.texto(
        _IZQ - 52,
        y + alto / 2,
        unidad,
        "eje",
        text_anchor="middle",
        transform=f"rotate(-90 {_IZQ - 52} {y + alto / 2})",
    )
    if capacidad is not None:
        valor, rotulo = capacidad
        lienzo.add(
            f'<line x1="{_IZQ}" x2="{_IZQ + lienzo.ancho}" y1="{py(valor)}" '
            f'y2="{py(valor)}" class="capacidad"/>'
        )
        lienzo.texto(_IZQ + lienzo.ancho + 8, py(valor) + 4, rotulo, "eje")
    for clase, puntos in series:
        if not puntos:
            continue
        trazo = " ".join(f"{lienzo.x(t)},{py(v)}" for t, v in puntos)
        if clase == "area":
            base = py(0)
            lienzo.add(
                f'<polygon class="relleno" points="{lienzo.x(puntos[0][0])},{base} '
                f'{trazo} {lienzo.x(puntos[-1][0])},{base}"/>'
            )
        lienzo.add(f'<polyline class="{clase}" points="{trazo}"/>')


def _series_recursos(
    recursos: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[tuple[float, float]]], dict[int, Any]]:
    """Agrega las muestras: totales por instante y RSS por proceso."""
    host: dict[str, Any] = {}
    por_instante: dict[float, dict[str, float]] = defaultdict(
        lambda: {"cpu": 0.0, "mem": 0.0}
    )
    sistema_cpu, sistema_mem = [], []
    procesos: dict[int, dict[str, Any]] = {}
    for fila in recursos:
        rol = fila["role"]
        if rol == "host":
            host = fila
            continue
        t = _epoch(fila["at"])
        if rol == "system":
            sistema_cpu.append((t, fila["cpu_cores"]))
            sistema_mem.append((t, fila["mem_mb"] / 1024))
            continue
        total = por_instante[t]
        total["cpu"] += fila["cpu_cores"]
        total["mem"] += fila["mem_mb"] / 1024
        proceso = procesos.setdefault(fila["pid"], {"role": rol, "puntos": []})
        proceso["puntos"].append((t, fila["mem_mb"] / 1024))
    instantes = sorted(por_instante)
    series = {
        "cpu": [(t, por_instante[t]["cpu"]) for t in instantes],
        "mem": [(t, por_instante[t]["mem"]) for t in instantes],
        "sistema_cpu": sistema_cpu,
        "sistema_mem": sistema_mem,
    }
    return host, series, procesos


_CLARO = {
    "fondo": "#fcfcfb",
    "tinta": "#0b0b0b",
    "tinta_2": "#52514e",
    "tinta_3": "#8a8984",
    "rejilla": "#e4e3df",
    "banda": "#f0efec",
    "serie": "#2a78d6",
    "modelos": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
}
_OSCURO = {
    "fondo": "#1a1a19",
    "tinta": "#ffffff",
    "tinta_2": "#c3c2b7",
    "tinta_3": "#8f8e86",
    "rejilla": "#33332f",
    "banda": "#262624",
    "serie": "#3987e5",
    "modelos": ("#3987e5", "#d95926", "#199e70", "#c98500"),
}


def _reglas(p: dict[str, Any]) -> str:
    """Colores explicitos por clase (sin variables CSS, que no todo visor lee)."""
    return (
        f".fondo{{fill:{p['fondo']}}}"
        f"text{{fill:{p['tinta_2']}}}"
        f".titulo,.titulo-panel,.etapa-texto{{fill:{p['tinta']}}}"
        f".rejilla{{stroke:{p['rejilla']}}}"
        f".capacidad,.division{{stroke:{p['tinta_3']}}}"
        f".area,.principal{{stroke:{p['serie']}}}"
        f".relleno{{fill:{p['serie']}}}"
        f".sistema,.worker{{stroke:{p['tinta_3']}}}"
        f".banda{{fill:{p['banda']}}}"
        f".fallida{{stroke:{p['tinta']}}}"
        f".raya{{fill:{p['tinta_2']}}}"
        + "".join(f".m{i}{{fill:{color}}}" for i, color in enumerate(p["modelos"]))
    )


def _estilo() -> str:
    return (
        "text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;"
        "font-size:11px}"
        ".titulo{font-size:17px;font-weight:600}"
        ".subtitulo{font-size:12px}"
        ".titulo-panel{font-size:12.5px;font-weight:600}"
        ".etapa-texto{font-weight:600}"
        ".rejilla{stroke-width:1}"
        ".capacidad{stroke-width:1.5;stroke-dasharray:5 4}"
        ".area{fill:none;stroke-width:2;stroke-linejoin:round}"
        ".relleno{fill-opacity:.16}"
        ".sistema{fill:none;stroke-width:1.5;stroke-linejoin:round}"
        ".principal{fill:none;stroke-width:2}"
        ".worker{fill:none;stroke-width:1;stroke-opacity:.7}"
        ".division{stroke-width:1;stroke-dasharray:2 3}"
        ".fallida{fill:url(#rayado);stroke-width:1}"
        + _reglas(_CLARO)
        + f"@media (prefers-color-scheme: dark){{{_reglas(_OSCURO)}}}"
    )


def render_run_chart(run_dir: str | Path, output: str | Path | None = None) -> Path:
    """Dibuja ``telemetria.svg`` (u ``output``) a partir de una corrida terminada."""
    directorio = Path(run_dir)
    if not (directorio / "unidades.jsonl").exists():
        raise FileNotFoundError(f"{directorio} no tiene unidades.jsonl")
    unidades = _leer_jsonl(directorio / "unidades.jsonl")
    corrida_path = directorio / "corrida.json"
    corrida = (
        json.loads(corrida_path.read_text(encoding="utf-8"))
        if corrida_path.exists()
        else {}
    )
    host, series, procesos = _series_recursos(
        _leer_jsonl(directorio / "recursos.jsonl")
    )
    etapas = [
        (etapa, _epoch(tramo["started_at"]), _epoch(tramo["finished_at"]))
        for etapa, tramo in corrida.get("stage_times", {}).items()
    ]
    instantes = [t for _, a, b in etapas for t in (a, b)]
    instantes += [_epoch(u[c]) for u in unidades for c in ("started_at", "finished_at")]
    instantes += [t for t, _ in series["cpu"]]
    if not instantes:
        raise ValueError(f"{directorio} no tiene unidades ni muestras que graficar")
    lienzo = _Lienzo(min(instantes), max(instantes))

    carriles: dict[int, int] = {}
    for unidad in sorted(unidades, key=lambda u: u["started_at"]):
        carriles.setdefault(unidad["pid"], len(carriles))
    hay_recursos = bool(series["cpu"])
    y_etapas = 92.0
    y = y_etapas + 22 + _SEPARACION
    paneles_y = []
    if hay_recursos:
        paneles_y = [y + i * (_PANEL + _SEPARACION) for i in range(3)]
        y = paneles_y[-1] + _PANEL + _SEPARACION
    y_gantt = y
    alto_gantt = max(len(carriles), 1) * (_CARRIL + _HUECO)
    y_eje = y_gantt + alto_gantt + 8
    alto = y_eje + 54

    # Bandas de etapa, detras de todo.
    for indice, (etapa, inicio, fin) in enumerate(etapas):
        x0, x1 = lienzo.x(inicio), lienzo.x(fin)
        if indice % 2 == 0:
            lienzo.add(
                f'<rect class="banda" x="{x0}" y="{y_etapas}" '
                f'width="{max(x1 - x0, 1)}" height="{y_eje - y_etapas}"/>'
            )
        lienzo.add(
            f'<line class="division" x1="{x0}" x2="{x0}" y1="{y_etapas}" y2="{y_eje}"/>'
        )
        etiqueta = f"{etapa} · {_duracion(fin - inicio)}"
        if x1 - x0 > 7 * len(etiqueta):
            lienzo.texto(
                (x0 + x1) / 2,
                y_etapas + 15,
                etiqueta,
                "etapa-texto",
                text_anchor="middle",
            )
        elif x1 - x0 > 20:
            lienzo.texto(
                (x0 + x1) / 2, y_etapas + 15, etapa, "etapa-texto", text_anchor="middle"
            )
    lienzo.texto(_IZQ - 8, y_etapas + 15, "Etapas", "eje", text_anchor="end")

    if hay_recursos:
        nucleos = float(host.get("cpu_cores") or 0)
        total_gb = float(host.get("mem_mb") or 0) / 1024
        maximo_cpu = max(
            [nucleos] + [v for _, v in series["cpu"] + series["sistema_cpu"]]
        )
        _panel(
            lienzo,
            paneles_y[0],
            "CPU",
            "núcleos",
            maximo_cpu,
            [
                ("sistema", series["sistema_cpu"]),
                ("area", series["cpu"]),
            ],
            capacidad=(nucleos, f"{nucleos:g} núcleos") if nucleos else None,
            leyenda=[("area", "procesos de la corrida"), ("sistema", "máquina")],
        )
        maximo_mem = max(
            [total_gb] + [v for _, v in series["mem"] + series["sistema_mem"]]
        )
        _panel(
            lienzo,
            paneles_y[1],
            "Memoria",
            "GB",
            maximo_mem,
            [
                ("sistema", series["sistema_mem"]),
                ("area", series["mem"]),
            ],
            capacidad=(total_gb, f"{total_gb:.0f} GB total") if total_gb else None,
            leyenda=[
                ("area", "RSS de la corrida"),
                ("sistema", "en uso en la máquina"),
            ],
        )
        por_proceso = sorted(procesos.values(), key=lambda p: p["role"] == "main")
        _panel(
            lienzo,
            paneles_y[2],
            "Memoria por proceso",
            "GB (RSS)",
            max((v for p in por_proceso for _, v in p["puntos"]), default=0.0),
            [
                ("principal" if p["role"] == "main" else "worker", p["puntos"])
                for p in por_proceso
            ],
            leyenda=[
                ("principal", "proceso principal"),
                ("worker", "cada proceso del pool"),
            ],
        )

    # Traza de modelos: una fila por proceso, una barra por unidad.
    lienzo.texto(
        _IZQ, y_gantt - 14, "Unidades SKU × familia por proceso", "titulo-panel"
    )
    x_leyenda = _IZQ + 260
    presentes = {u.get("model") for u in unidades}
    for indice, modelo in enumerate(_MODELOS):
        if modelo not in presentes:
            continue
        lienzo.add(
            f'<rect class="m{indice}" x="{x_leyenda}" y="{y_gantt - 23}" '
            f'width="12" height="10" rx="2"/>'
        )
        lienzo.texto(x_leyenda + 17, y_gantt - 14, modelo, "eje")
        x_leyenda += 30 + 7 * len(modelo)
    if any(u["state"] != "completada" for u in unidades):
        lienzo.add(
            f'<rect class="fallida" x="{x_leyenda}" y="{y_gantt - 23}" '
            f'width="12" height="10"/>'
        )
        lienzo.texto(x_leyenda + 17, y_gantt - 14, "fallida", "eje")
    for pid, carril in carriles.items():
        yc = y_gantt + carril * (_CARRIL + _HUECO)
        rotulo = "sin proceso" if pid == 0 else f"pid {pid}"
        lienzo.texto(_IZQ - 8, yc + _CARRIL - 1, rotulo, "eje", text_anchor="end")
    for unidad in unidades:
        yc = y_gantt + carriles[unidad["pid"]] * (_CARRIL + _HUECO)
        x0 = lienzo.x(_epoch(unidad["started_at"]))
        x1 = lienzo.x(_epoch(unidad["finished_at"]))
        modelo = unidad.get("model")
        clase = (
            "fallida"
            if unidad["state"] != "completada"
            else f"m{_MODELOS.index(modelo)}"
            if modelo in _MODELOS
            else "raya"
        )
        cpu = unidad.get("cpu_s")
        detalle = (
            f"{unidad['stage']} · SKU {unidad['sku_id']} · {modelo} · "
            f"{unidad['seconds']:.1f} s"
            + (f" · CPU {cpu:.1f} s" if cpu is not None else "")
            + f" · {unidad['state']}"
            + (f" · {unidad['error']}" if unidad.get("error") else "")
        )
        lienzo.add(
            f'<rect class="{clase}" x="{x0}" y="{yc}" width="{max(x1 - x0 - 2, 1)}" '
            f'height="{_CARRIL}" rx="1.5"><title>{escape(detalle)}</title></rect>'
        )

    # Eje de tiempo comun.
    paso = _paso_tiempo(lienzo.duracion)
    marca = 0.0
    while marca <= lienzo.duracion + 1e-9:
        x = lienzo.x(lienzo.t0 + marca)
        lienzo.add(
            f'<line class="rejilla" x1="{x}" x2="{x}" y1="{y_eje}" y2="{y_eje + 5}"/>'
        )
        lienzo.texto(x, y_eje + 18, _reloj(marca), "eje", text_anchor="middle")
        marca += paso
    lienzo.texto(
        _IZQ + lienzo.ancho / 2,
        y_eje + 38,
        "tiempo desde el inicio (min:s)",
        "eje",
        text_anchor="middle",
    )

    run_id = corrida.get("run_id", directorio.name)
    fallidas = sum(u["state"] != "completada" for u in unidades)
    partes = [_duracion(lienzo.duracion), f"{len(unidades)} unidades"]
    if fallidas:
        partes[-1] += f" ({fallidas} fallidas)"
    partes.append(f"{sum(1 for p in carriles if p)} procesos con unidades")
    telemetria = corrida.get("telemetry") or {}
    if telemetria.get("peak_cpu_cores") is not None:
        partes.append(f"pico CPU {telemetria['peak_cpu_cores']:g} núcleos")
        partes.append(f"pico RSS {telemetria['peak_rss_mb'] / 1024:.1f} GB")
    encabezado = (
        f'<text x="{_IZQ}" y="34" class="titulo">Corrida {escape(run_id)}</text>'
        f'<text x="{_IZQ}" y="56" class="subtitulo">{escape(" · ".join(partes))}'
        "</text>"
    )
    return _escribir_svg(
        Path(output) if output is not None else directorio / CHART_NAME,
        alto,
        f"Corrida {run_id}",
        encabezado + "".join(lienzo.partes),
    )


def _escribir_svg(ruta: Path, alto: float, titulo: str, cuerpo: str) -> Path:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_ANCHO}" '
        f'height="{alto:.0f}" viewBox="0 0 {_ANCHO} {alto:.0f}" role="img">'
        f"<title>{escape(titulo)}</title>"
        f"<style>{_estilo()}</style>"
        '<defs><pattern id="rayado" width="4" height="4" '
        'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<rect class="raya" width="2" height="4"/></pattern></defs>'
        f'<rect class="fondo" width="100%" height="100%"/>'
        f"{cuerpo}</svg>\n"
    )
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(svg, encoding="utf-8")
    return ruta
