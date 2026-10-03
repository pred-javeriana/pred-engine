"""Grafico de una corrida: etapas, recursos y una barra por unidad, en SVG valido."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pred_engine.cli import main
from pred_engine.run_chart import render_run_chart

_SVG = "{http://www.w3.org/2000/svg}"
_T0 = 1_790_000_000.0


def _iso(segundos: float) -> str:
    return datetime.fromtimestamp(_T0 + segundos, UTC).isoformat()


def _unidad(stage, sku, model, pid, inicio, fin, state="completada", error=None):
    return {
        "stage": stage,
        "sku_id": sku,
        "family": {"sarima": "classical", "lightgbm": "ml"}[model],
        "model": model,
        "state": state,
        "pid": pid,
        "started_at": _iso(inicio),
        "finished_at": _iso(fin),
        "seconds": fin - inicio,
        "cpu_s": (fin - inicio) * 0.9,
        "error": error,
    }


def _corrida(directorio: Path, *, recursos: bool = True) -> Path:
    directorio.mkdir(parents=True)
    unidades = [
        _unidad("L2", "A", "sarima", 11, 1, 20),
        _unidad("L2", "A", "lightgbm", 12, 1, 8),
        _unidad("L2", "B", "lightgbm", 12, 8, 9, "fallida", "ValueError: sin trials"),
        _unidad("L3", "A", "sarima", 21, 21, 25),
    ]
    (directorio / "unidades.jsonl").write_text(
        "".join(json.dumps(u) + "\n" for u in unidades)
    )
    corrida = {
        "run_id": "r-prueba",
        "stage_times": {
            "L1": {"started_at": _iso(0), "finished_at": _iso(1), "seconds": 1},
            "L2": {"started_at": _iso(1), "finished_at": _iso(20), "seconds": 19},
            "L3": {"started_at": _iso(20), "finished_at": _iso(26), "seconds": 6},
        },
    }
    if recursos:
        filas = [
            {"at": _iso(0), "role": "host", "pid": None, "cpu_cores": 4, "mem_mb": 8192}
        ]
        for t in range(27):
            pool = [11, 12] if t <= 20 else [21]
            filas.append(
                {
                    "at": _iso(t),
                    "role": "system",
                    "pid": None,
                    "cpu_cores": 2.5,
                    "mem_mb": 3000.0,
                }
            )
            filas.append(
                {
                    "at": _iso(t),
                    "role": "main",
                    "pid": 10,
                    "cpu_cores": 0.1,
                    "mem_mb": 300.0,
                }
            )
            filas += [
                {
                    "at": _iso(t),
                    "role": "worker",
                    "pid": pid,
                    "cpu_cores": 1.0,
                    "mem_mb": 200.0 + t,
                }
                for pid in pool
            ]
        (directorio / "recursos.jsonl").write_text(
            "".join(json.dumps(f) + "\n" for f in filas)
        )
        corrida["telemetry"] = {"peak_cpu_cores": 2.1, "peak_rss_mb": 740.0}
    (directorio / "corrida.json").write_text(json.dumps(corrida))
    return directorio


def _textos(raiz: ET.Element) -> list[str]:
    return [t.text or "" for t in raiz.iter(f"{_SVG}text")]


def test_dibuja_etapas_recursos_y_una_barra_por_unidad(tmp_path):
    ruta = render_run_chart(_corrida(tmp_path / "r-prueba"))

    assert ruta == tmp_path / "r-prueba" / "telemetria.svg"
    raiz = ET.parse(ruta).getroot()
    textos = _textos(raiz)
    assert "Corrida r-prueba" in textos
    assert any("4 unidades (1 fallidas)" in t for t in textos)
    assert any("pico CPU 2.1 núcleos" in t for t in textos)
    for rotulo in ("CPU", "Memoria", "Memoria por proceso", "4 núcleos", "8 GB total"):
        assert rotulo in textos
    assert {"sarima", "lightgbm", "fallida"} <= set(textos)
    assert any(t.startswith("L2 · 19") for t in textos)
    assert {"pid 11", "pid 12", "pid 21"} <= set(textos)

    barras = [r for r in raiz.iter(f"{_SVG}rect") if r.find(f"{_SVG}title") is not None]
    assert len(barras) == 4
    detalles = [b.find(f"{_SVG}title").text for b in barras]
    assert detalles[0] == "L2 · SKU A · sarima · 19.0 s · CPU 17.1 s · completada"
    assert "fallida · ValueError: sin trials" in detalles[2]
    assert [b.get("class") for b in barras] == ["m0", "m1", "fallida", "m0"]
    # Una linea por proceso en "Memoria por proceso", la del principal incluida.
    assert (
        len(
            [
                p
                for p in raiz.iter(f"{_SVG}polyline")
                if p.get("class") in ("worker", "principal")
            ]
        )
        == 4
    )


def test_sin_muestras_de_recursos_dibuja_solo_etapas_y_unidades(tmp_path):
    destino = tmp_path / "otro" / "grafico.svg"
    ruta = render_run_chart(_corrida(tmp_path / "r", recursos=False), destino)

    assert ruta == destino
    raiz = ET.parse(ruta).getroot()
    assert "CPU" not in _textos(raiz)
    assert not list(raiz.iter(f"{_SVG}polyline"))
    assert (
        len([r for r in raiz.iter(f"{_SVG}rect") if r.find(f"{_SVG}title") is not None])
        == 4
    )


def test_una_corrida_sin_unidades_es_un_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="unidades.jsonl"):
        render_run_chart(tmp_path)


def test_cli_telemetry_escribe_el_svg_y_reporta_errores(tmp_path, capsys):
    directorio = _corrida(tmp_path / "r-prueba")
    assert main(["telemetry", str(directorio)]) == 0
    assert capsys.readouterr().out.strip() == str(directorio / "telemetria.svg")

    salida = tmp_path / "g.svg"
    assert main(["telemetry", str(directorio), "--output", str(salida)]) == 0
    assert salida.read_text().startswith("<svg")

    assert main(["telemetry", str(tmp_path / "no-existe")]) == 1
    assert capsys.readouterr().err.startswith("error: ")
