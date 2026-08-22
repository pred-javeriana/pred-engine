"""Prompt zero-shot bilingue anclado al contrato canonico y al dataset de proyecto."""

from __future__ import annotations

import json

import pandas as pd

from pred_engine.comun.modelos import CANONICAL_FIELDS

N_SAMPLE_ROWS = 5


def sample_header_frame(
    frame: pd.DataFrame, n_rows: int = N_SAMPLE_ROWS
) -> pd.DataFrame:
    """Primeras n filas; no muta el marco original."""
    if frame.empty:
        raise ValueError("No se puede sondear un DataFrame vacio")
    return frame.head(n_rows).copy()


def build_alignment_prompt(frame: pd.DataFrame, n_rows: int = N_SAMPLE_ROWS) -> str:
    """Inyecta cabeceras + muestra. El LLM debe devolver solo JSON del contrato."""
    muestra = sample_header_frame(frame, n_rows)
    cabeceras = [str(c).strip() for c in frame.columns]
    contrato = ", ".join(CANONICAL_FIELDS)
    return (
        "You are the PRED demand-forecasting ingestion aligner.\n"
        "Eres el alineador de ingesta de PRED (pronostico de demanda).\n\n"
        "Map each canonical field to EXACTLY one source column name "
        "from the header list.\n"
        f"Canonical fields: {contrato}\n\n"
        "Definitions / definiciones:\n"
        "- sku_id: unique product identifier (SKU, item id, codigo de producto). "
        "NOT the product name, NOT the vendor id.\n"
        "- timestamp: observation/transaction date. NOT a lead time.\n"
        "- demand_qty: quantity DEMANDED, USED, SOLD or CONSUMED in that period "
        "(usage, avg usage per day, unidades, demanda). "
        "NEVER map on-hand stock, current stock, inventory level, min/max capacity, "
        "unit cost, or vendor id to demand_qty. "
        "NUNCA uses Current_Stock / existencias / stock on hand como demanda.\n"
        "- lead_time_days: replenishment lead time in DAYS (restock lead time, plazo). "
        "NOT a vendor id.\n\n"
        "Rules:\n"
        "1. Return a single JSON object with keys exactly: "
        f"{json.dumps(list(CANONICAL_FIELDS))}.\n"
        "2. Values are the exact source column names, or null if you cannot map "
        "with high confidence. Do NOT invent column names.\n"
        "3. Each source column maps to at most one canonical field.\n"
        "4. Extra source columns are ignored (they will be dropped).\n"
        "5. If this file is not demand/inventory time-series data, return nulls.\n\n"
        f"Source headers: {json.dumps(cabeceras, ensure_ascii=False)}\n\n"
        "First rows (CSV):\n"
        f"{muestra.to_csv(index=False)}"
    )
