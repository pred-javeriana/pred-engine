"""Reconstruye `timing22.csv`, la muestra real del nivel `muestra22`.

Uso: `uv run python .github/scripts/muestra22.py PANEL_LISTO.csv SALIDA.csv`.
Quita los SKU sin demanda positiva hasta t* (la puerta L1.4 del motor) y toma
22 SKU al azar con semilla 17. Imprime el id de la muestra que lleva el recibo;
con el panel preparado del equipo da `m22-941198f5af6e`.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

from pred_engine.comun.reserva import ReserveCut

N_SKUS = 22
SEMILLA = 17


def main(origen: str, destino: str) -> None:
    panel = pd.read_csv(origen, dtype={"sku_id": str})
    fechas = pd.to_datetime(panel["timestamp"])
    corte = ReserveCut.of(panel.assign(timestamp=fechas))
    con_demanda = panel.loc[
        (fechas <= corte.t_star) & (panel["demand_qty"] > 0), "sku_id"
    ]
    elegibles = pd.Series(sorted(set(con_demanda)))
    elegidos = set(elegibles.sample(N_SKUS, random_state=SEMILLA))
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    panel[panel["sku_id"].isin(elegidos)].to_csv(destino, index=False)
    with open(destino, "rb") as archivo:
        print(f"m22-{hashlib.sha256(archivo.read()).hexdigest()[:12]}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
