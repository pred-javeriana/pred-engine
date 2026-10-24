"""Verifica los artefactos de una corrida de `pred-engine run` en CI.

Uso: `python .github/scripts/verificar_corrida.py RUN_DIR DB_PATH`. Cada falla
nombra el archivo de la corrida que la produjo y se informa como anotacion de
GitHub Actions; el codigo de salida es 1 si hubo alguna.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import xml.etree.ElementTree as ET
from contextlib import closing
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from pred_engine.comun.modelos.manifiesto_candidatos import ContextoParticion
from pred_engine.forecasting.adaptador_candidatos import validar_manifiesto
from pred_engine.forecasting.persistencia_evidencia import VERSION_ESQUEMA_M3

ARCHIVOS = (
    "corrida.json",
    "candidatos.json",
    "pronosticos.parquet",
    "evaluacion.parquet",
    "walk_forward.parquet",
    "unidades.jsonl",
    "recursos.jsonl",
    "telemetria.svg",
)


def verificar(run_dir: Path, db_path: Path) -> list[str]:
    faltan = [
        f"{nombre}: falta o esta vacio"
        for nombre in ARCHIVOS
        if not (run_dir / nombre).is_file() or (run_dir / nombre).stat().st_size == 0
    ]
    if faltan:
        return faltan
    errores: list[str] = []

    def exigir(condicion: bool, mensaje: str) -> None:
        if not condicion:
            errores.append(mensaje)

    corrida = json.loads((run_dir / "corrida.json").read_text())
    resumen = corrida["summary"]
    estados = {s["id"]: s["state"] for s in resumen["stages"]}
    exigir(
        estados
        == {"L1": "completed", "L2": "completed", "L3": "completed", "L4": "completed"},
        f"corrida.json: estados de etapa inesperados {estados}",
    )
    exigir(resumen["complete"], "corrida.json: pipeline incompleto")
    m3 = resumen.get("m3", {})
    exigir(m3.get("estado") == "PUBLICADA", "corrida.json: M3 no publicada")
    if not m3:
        return errores
    try:
        conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return errores + [f"base central: {exc}"]
    with closing(conn):
        exigir(
            conn.execute("PRAGMA user_version").fetchone()[0] == VERSION_ESQUEMA_M3,
            "base central: version incompatible",
        )
        exigir(
            not conn.execute("PRAGMA foreign_key_check").fetchall(),
            "base central: referencias rotas",
        )
        run_id = m3["run_id_m3"]
        exigir(
            conn.execute(
                "SELECT estado, datos_sinteticos FROM m3_corridas WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            == ("PUBLICADA", 1),
            "base central: falta publicacion sintetica M3",
        )
        previstas = conn.execute(
            "SELECT n_unidades_previstas FROM m3_protocolos WHERE run_id = ?", (run_id,)
        ).fetchone()
        total = conn.execute(
            "SELECT COUNT(*) FROM m3_unidades WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        exigir(
            previstas == (total,) and total == m3["n_unidades"] and total > 0,
            "base central: unidades incompletas o sin protocolo",
        )
        horizonte = m3["horizonte"]
        exigir(
            not conn.execute(
                "SELECT 1 FROM m3_pronosticos WHERE run_id = ? "
                "GROUP BY sku, candidato_id, origen "
                "HAVING COUNT(*) != ? OR MIN(h) != 1 OR MAX(h) != ?",
                (run_id, horizonte, horizonte),
            ).fetchall(),
            "base central: horizontes parciales",
        )
        exigir(
            conn.execute(
                "SELECT COUNT(*) FROM m3_pronosticos WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            == total * horizonte,
            "base central: faltan pronosticos",
        )
        exigir(
            conn.execute(
                "SELECT COUNT(*) FROM m3_veredictos WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            > 1,
            "base central: faltan veredictos multi-SKU",
        )
        exigir(
            not conn.execute(
                "SELECT 1 FROM m3_veredictos WHERE run_id = ? "
                "AND (dm_p_valor IS NOT NULL OR dm_p_ajustado IS NOT NULL)",
                (run_id,),
            ).fetchall(),
            "base central: inferencia sobre datos sinteticos",
        )
        exigir(bool(m3["archivos"]), "corrida.json: faltan exportaciones M3")
        for tabla, ruta in m3["archivos"].items():
            quoted = '"' + tabla.replace('"', '""') + '"'
            filas = conn.execute(
                f"SELECT COUNT(*) FROM {quoted} WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            exigir(
                len(pd.read_parquet(ruta)) == filas,
                f"{tabla}.parquet: difiere de SQLite",
            )
            exigir(
                len(pd.read_csv(m3["csv"][tabla])) == filas,
                f"{tabla}.csv: difiere de SQLite",
            )
    unidades = resumen["units"]
    exigir(unidades["L2"]["completed"] > 0, "corrida.json: ninguna unidad L2 completa")
    exigir(unidades["L3"]["completed"] > 0, "corrida.json: ninguna unidad L3 completa")
    exigir(
        corrida["inputs"]["m1"]["skus"] > 1,
        f"corrida.json: M1 publico {corrida['inputs']['m1']['skus']} SKU; "
        "se esperan varios",
    )

    panel = pd.read_parquet(corrida["inputs"]["m1"]["parquet"])
    try:
        handoff = validar_manifiesto(
            (run_dir / "candidatos.json").read_text(),
            esperado=ContextoParticion(
                ingesta_ref_m1=corrida["inputs"]["m1"]["sha256"],
                t_corte_reserva=date.fromisoformat(resumen["reserve"]["t_star"]),
                fraccion_reserva=resumen["reserve"]["fraction"],
            ),
            skus_panel=set(panel["sku_id"]),
        )
    except Exception as exc:
        return errores + [f"candidatos.json: M3 rechaza el manifiesto: {exc}"]
    exigir(
        handoff.fallos == (),
        f"candidatos.json: M3 rechaza candidatos {handoff.fallos}",
    )
    familias = {c.familia for c in handoff.candidatos}
    pedidas = set(corrida["settings"]["families"])
    exigir(
        familias == pedidas,
        f"candidatos.json: familias sin candidato {sorted(pedidas - familias)}",
    )
    exigir(
        len({c.sku for c in handoff.candidatos}) > 1,
        "candidatos.json: los candidatos cubren un solo SKU",
    )
    exigir(
        len(handoff.candidatos) == unidades["L2"]["completed"],
        f"candidatos.json: {len(handoff.candidatos)} candidatos para "
        f"{unidades['L2']['completed']} unidades L2 completas",
    )

    pronosticos = pd.read_parquet(run_dir / "pronosticos.parquet")
    esperadas = resumen["reserve"]["reserved_days"] * len(handoff.candidatos)
    exigir(
        len(pronosticos) == esperadas,
        f"pronosticos.parquet: {len(pronosticos)} filas; se esperan {esperadas} "
        "(dias reservados x candidatos)",
    )
    valores = pronosticos["pronostico"].to_numpy(dtype=float)
    exigir(
        bool(np.isfinite(valores).all() and (valores >= 0).all()),
        "pronosticos.parquet: hay pronosticos negativos o no finitos",
    )
    evaluacion = pd.read_parquet(run_dir / "evaluacion.parquet")
    exigir(
        len(evaluacion) == unidades["L3"]["completed"],
        f"evaluacion.parquet: {len(evaluacion)} filas para "
        f"{unidades['L3']['completed']} unidades L3 completas",
    )
    exigir(
        len(pd.read_parquet(run_dir / "walk_forward.parquet")) > 0,
        "walk_forward.parquet: sin ventanas",
    )

    with (run_dir / "unidades.jsonl").open() as archivo:
        traza = [json.loads(linea) for linea in archivo]
    total = unidades["L2"]["total"] + unidades["L3"]["total"]
    exigir(
        len(traza) == total,
        f"unidades.jsonl: {len(traza)} lineas para {total} unidades L2 y L3",
    )
    with (run_dir / "recursos.jsonl").open() as archivo:
        roles = {json.loads(linea)["role"] for linea in archivo}
    exigir("worker" in roles, "recursos.jsonl: sin muestras de los procesos del pool")
    estudios = list((run_dir / "hpo").glob("*/manifiesto.json"))
    exigir(
        len(estudios) == unidades["L2"]["total"],
        f"hpo/: {len(estudios)} estudios para {unidades['L2']['total']} unidades L2",
    )

    try:
        raiz = ET.parse(run_dir / "telemetria.svg").getroot()
    except ET.ParseError as exc:
        return errores + [f"telemetria.svg: XML invalido: {exc}"]
    exigir(raiz.tag.endswith("svg"), f"telemetria.svg: raiz {raiz.tag}, no svg")
    return errores


def main() -> int:
    run_dir = Path(sys.argv[1])
    errores = verificar(run_dir, Path(sys.argv[2]))
    for error in errores:
        print(f"::error title=Corrida {run_dir.name}::{error}")
    if not errores:
        print(f"Corrida {run_dir.name}: artefactos verificados")
    return 1 if errores else 0


if __name__ == "__main__":
    raise SystemExit(main())
