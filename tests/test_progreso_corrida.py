"""La corrida deja su flujo de progreso, coherente con unidades.jsonl (G3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pred_engine.cli import main

_ARGS = [
    "--trials", "1", "--min-train", "40", "--horizon", "2", "--step", "14",
    "--metric", "mae", "--seed", "17", "--families", "classical", "--workers", "2",
]  # fmt: skip


def _csv(raiz: Path, *, corto: bool = False) -> Path:
    """Dos SKU; con ``corto``, sku-b tiene 30 dias, termina antes de t* y se excluye."""
    n = 120
    filas = []
    for sku, largo in (("sku-a", n), ("sku-b", 30 if corto else n)):
        for i in range(largo):
            filas.append(
                {
                    "sku_id": sku,
                    "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                    "demand_qty": 20 + 0.35 * i + (i % 5 - 2),
                    "lead_time_days": 3,
                }
            )
    ruta = raiz / "ventas.csv"
    pd.DataFrame(filas).to_csv(ruta, index=False)
    return ruta


def _correr(
    raiz: Path, capsys: pytest.CaptureFixture[str], **kw: Any
) -> dict[str, Any]:
    csv = _csv(raiz, **kw)
    main(["run", "--csv", str(csv), "--data-root", str(raiz / "data"), *_ARGS])
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def _eventos(carpeta: Path) -> list[dict[str, Any]]:
    texto = (carpeta / "progreso.jsonl").read_text(encoding="utf-8")
    return [json.loads(linea) for linea in texto.splitlines()]


def test_la_corrida_deja_un_flujo_completo_y_coherente(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    resumen = _correr(tmp_path, capsys)
    carpeta = Path(resumen["run_dir"])
    eventos = _eventos(carpeta)

    assert "progreso.jsonl" in resumen["files"]
    assert json.loads((carpeta / "corrida.json").read_text())["files"]["progreso"] == (
        "progreso.jsonl"
    )
    assert eventos[0]["event"] == "run_started"
    assert eventos[0]["schema"] == "pred-engine.progreso/1"
    assert eventos[0]["attempt"] == 1 and eventos[0]["run_id"] == resumen["run_id"]
    assert eventos[-1]["event"] == "run_finished"
    # L4 no existe: la corrida termina sin fallos y con el codigo de salida 7.
    assert (eventos[-1]["state"], eventos[-1]["exit_code"]) == ("completed", 7)

    etapas = [
        (e["stage"], e["event"]) for e in eventos if e["event"].startswith("stage_")
    ]
    assert etapas[:2] == [("L1", "stage_started"), ("L1", "stage_finished")]
    planes = {e["stage"]: e["units"] for e in eventos if e["event"] == "planned"}
    assert set(planes) == {"L2", "L3"}

    # Cada unidad planeada termina una vez y coincide con unidades.jsonl.
    finales = {
        (e["stage"], e["sku_id"], e["family"]): e["state"]
        for e in eventos
        if e["event"] == "unit_finished"
    }
    planeadas = {(s, u["sku_id"], u["family"]) for s, us in planes.items() for u in us}
    assert set(finales) == planeadas
    unidades = [
        json.loads(linea)
        for linea in (carpeta / "unidades.jsonl").read_text().splitlines()
    ]
    assert {
        (u["stage"], u["sku_id"], u["family"]): u["state"] for u in unidades
    } == finales
    iniciadas = [e for e in eventos if e["event"] == "unit_started"]
    assert len(iniciadas) == len(planeadas)


def test_un_sku_sin_historia_suficiente_se_anota_como_excluido(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    carpeta = Path(_correr(tmp_path, capsys, corto=True)["run_dir"])

    excluidos = [e for e in _eventos(carpeta) if e["event"] == "excluded"]
    assert [e["sku_id"] for e in excluidos] == ["sku-b"]
    assert "termina antes de t*" in excluidos[0]["cause"]


def test_repetir_la_corrida_agrega_un_intento_sin_perder_el_anterior(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    primero = _correr(tmp_path, capsys)
    segundo = _correr(tmp_path, capsys)

    assert segundo["run_id"] == primero["run_id"]
    inicios = [
        e for e in _eventos(Path(segundo["run_dir"])) if e["event"] == "run_started"
    ]
    assert [e["attempt"] for e in inicios] == [1, 2]
