"""La corrida deja su flujo de progreso, coherente con unidades.jsonl (G3)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pred_engine.cli import main
from tests._plataforma import crear_base_plataforma
from tests._series import serie_ar_estacional

_ARGS = [
    "--trials", "1", "--min-train", "40", "--horizon", "2", "--step", "14",
    "--metric", "mae", "--seed", "15", "--families", "classical", "--workers", "2",
]  # fmt: skip


def _csv(raiz: Path, *, corto: bool = False) -> Path:
    """Dos SKU; con ``corto``, sku-b tiene 30 dias, termina antes de t* y se excluye."""
    n = 120
    demanda = np.round(serie_ar_estacional(n), 2)
    filas = []
    for sku, largo in (("sku-a", n), ("sku-b", 30 if corto else n)):
        for i in range(largo):
            filas.append(
                {
                    "sku_id": sku,
                    "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                    "demand_qty": demanda[i],
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
    if not (raiz / "data" / "pred.db").exists():
        crear_base_plataforma(raiz / "data" / "pred.db")
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
    # La evaluacion reservada queda publicada en la misma base de plataforma.
    assert (eventos[-1]["state"], eventos[-1]["exit_code"]) == ("completed", 0)
    assert resumen["m3"]["estado"] == "PUBLICADA"

    etapas = [
        (e["stage"], e["event"]) for e in eventos if e["event"].startswith("stage_")
    ]
    assert etapas[:2] == [("L1", "stage_started"), ("L1", "stage_finished")]
    planes = {e["stage"]: e["units"] for e in eventos if e["event"] == "planned"}
    assert set(planes) == {"L2", "L3", "L4"}

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
    # L4 todavia no anota unit_started ni entra a unidades.jsonl (bloque 4).
    finales_m2 = {k: v for k, v in finales.items() if k[0] != "L4"}
    assert {
        (u["stage"], u["sku_id"], u["family"]): u["state"] for u in unidades
    } == finales_m2
    iniciadas = [e for e in eventos if e["event"] == "unit_started"]
    assert len(iniciadas) == len(finales_m2)


def test_m3_anuncia_su_plan_y_cada_sku_guardado_en_el_progreso(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    resumen = _correr(tmp_path, capsys)
    m3 = [e for e in _eventos(Path(resumen["run_dir"])) if e.get("stage") == "L4"]

    # Mismo formato que M2: una unidad por SKU x candidato, con familia y modelo.
    esperadas = {("sku-a", "classical", "sarima"), ("sku-b", "classical", "sarima")}
    orden = [e["event"] for e in m3]
    assert (orden[0], orden[1], orden[-1]) == (
        "stage_started",
        "planned",
        "stage_finished",
    )
    (plan,) = [e for e in m3 if e["event"] == "planned"]
    assert {(u["sku_id"], u["family"], u["model"]) for u in plan["units"]} == esperadas
    fines = [e for e in m3 if e["event"] == "unit_finished"]
    assert sorted((e["sku_id"], e["family"], e["model"]) for e in fines) == sorted(
        esperadas
    )
    assert {e["state"] for e in fines} == {"completada"}


def test_un_sku_sin_historia_suficiente_se_anota_como_excluido(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    carpeta = Path(_correr(tmp_path, capsys, corto=True)["run_dir"])

    excluidos = [e for e in _eventos(carpeta) if e["event"] == "excluded"]
    assert [e["sku_id"] for e in excluidos] == ["sku-b"]
    assert "termina antes de t*" in excluidos[0]["cause"]
    assert excluidos[0]["code"] == "eligibility.no_reserved_observations"


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


def test_reintento_reparado_conserva_la_publicacion_con_fallo_tecnico(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from pred_engine.optimizacion.router import SelectionRouter

    identidad = [
        ("sku_id", pa.large_string()),
        ("sku_class", pa.large_string()),
        ("familia", pa.large_string()),
        ("modelo", pa.large_string()),
    ]
    esquemas = {
        "pronosticos": pa.schema(
            [
                *identidad,
                ("timestamp", pa.timestamp("ns")),
                ("pronostico", pa.float64()),
            ]
        ),
        "evaluacion": pa.schema(
            [
                *identidad,
                ("metrica", pa.large_string()),
                ("valor_agregado", pa.float64()),
                ("n_ventanas", pa.int64()),
                ("mae_media", pa.float64()),
                ("rmse_media", pa.float64()),
                ("smape_media", pa.float64()),
                ("mase_media", pa.float64()),
            ]
        ),
        "walk_forward": pa.schema(
            [
                ("sku_id", pa.large_string()),
                ("familia", pa.large_string()),
                ("ventana", pa.int64()),
                ("timestamp", pa.timestamp("ns")),
                ("real", pa.float64()),
                ("pronostico", pa.float64()),
            ]
        ),
    }

    def leer_exportaciones(reporte: dict[str, Any]) -> list[int]:
        filas = []
        for nombre, esquema in esquemas.items():
            tabla = pq.read_table(
                Path(reporte["run_dir"]) / f"{nombre}.parquet",
                columns=esquema.names,
            )
            assert tabla.schema.equals(esquema)
            filas.append(tabla.num_rows)
        return filas

    # L2 selecciona con execute_with_walk_forward (no con execute).
    ejecutar = SelectionRouter.execute_with_walk_forward
    averia = {"activa": True}

    def con_averia(router: SelectionRouter, request: Any, decision: Any) -> Any:
        if averia["activa"]:
            raise RuntimeError("fallo transitorio de L2")
        return ejecutar(router, request, decision)

    monkeypatch.setattr(SelectionRouter, "execute_with_walk_forward", con_averia)
    crear_base_plataforma(tmp_path / "data" / "pred.db")
    args = [
        "run", "--csv", str(_csv(tmp_path)), "--data-root", str(tmp_path / "data"),
        *_ARGS, "--workers", "1",
    ]  # fmt: skip

    def correr(*extra: str) -> tuple[int, dict[str, Any]]:
        codigo = main([*args, *extra])
        return codigo, json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    codigo, fallida = correr()
    assert codigo == 8
    assert {f["sku_id"] for f in fallida["failures"]} == {"sku-a", "sku-b"}
    assert fallida["m3"]["estado"] == "PUBLICADA"
    assert leer_exportaciones(fallida) == [0, 0, 0]

    # La causa se reparo, pero el mismo run_id reusa la publicacion congelada.
    averia["activa"] = False
    codigo, reintento = correr()
    assert codigo == 8
    assert reintento["failures"] == []
    assert reintento["m3"]["publicada_previamente"] is True
    assert reintento["m3"]["run_id_m3"] == fallida["m3"]["run_id_m3"]
    assert _eventos(Path(reintento["run_dir"]))[-1]["exit_code"] == 8
    with sqlite3.connect(tmp_path / "data" / "pred.db") as conn:
        veredictos = conn.execute(
            "SELECT DISTINCT veredicto FROM evidencia_publicada WHERE run_id = ?",
            (reintento["m3"]["run_id_m3"],),
        ).fetchall()
    assert veredictos == [("FALLO_TECNICO",)]

    # Un run_id nuevo produce resultados nuevos y sale limpio.
    codigo, nueva = correr("--run-id", "reparada")
    assert codigo == 0
    assert nueva["failures"] == []
    assert nueva["m3"]["run_id_m3"] != fallida["m3"]["run_id_m3"]
    assert leer_exportaciones(nueva) == [48, 2, 16]
