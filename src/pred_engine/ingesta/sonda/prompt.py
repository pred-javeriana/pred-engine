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
    """Inyecta cabeceras + muestra. El LLM devuelve diagnostico JSON, no mapeo."""
    muestra = sample_header_frame(frame, n_rows)
    cabeceras = [str(c).strip() for c in frame.columns]
    contrato = ", ".join(CANONICAL_FIELDS)
    return (
        "You are the PRED demand-forecasting ingestion header diagnostic assistant.\n"
        "Eres el asistente diagnostico de cabeceras de ingesta PRED.\n\n"
        "Compare source headers against the canonical contract. "
        "Do NOT rename columns. Do NOT modify data. "
        "Only emit a JSON diagnostic report for the human operator.\n"
        "Compara las cabeceras fuente contra el contrato. "
        "NO renombres columnas. NO modifiques datos. "
        "Solo emite un reporte JSON diagnostico para el operador humano.\n\n"
        f"Canonical contract / contrato canonico: {contrato}\n\n"
        "Definitions / definiciones:\n"
        "- sku_id: unique product identifier (SKU, item id, codigo de producto). "
        "NOT the product name, NOT the vendor id.\n"
        "- timestamp: observation/transaction date. NOT a lead time.\n"
        "- demand_qty: quantity DEMANDED, USED, SOLD or CONSUMED in that period "
        "(usage, avg usage per day, unidades, demanda). "
        "NEVER treat on-hand stock, current stock, inventory level, min/max capacity, "
        "unit cost, or vendor id as demand_qty. "
        "NUNCA uses Current_Stock / existencias / stock on hand como demanda.\n"
        "- lead_time_days: replenishment lead time in DAYS (restock lead time, plazo). "
        "NOT a vendor id.\n\n"
        "Output rules / reglas de salida:\n"
        '1. Return a single JSON object with keys exactly: "status", "diagnostic".\n'
        '2. "status" must be "accepted" only if ALL four canonical fields are present '
        "with exact column names matching the contract "
        f"({json.dumps(list(CANONICAL_FIELDS))}).\n"
        '3. "status" must be "rejected" if any canonical field is missing, misnamed, '
        "ambiguous, or semantically wrong (e.g. stock mapped as demand).\n"
        '4. "diagnostic" is an array of objects with keys: '
        '"field", "severity" ("error"|"info"), "message", "action".\n'
        "5. For rejected files, each missing or misnamed field must include "
        "an explicit "
        '"action" telling the operator how to rename or add columns '
        "(e.g. \"Renombrar la columna 'Date' a 'timestamp'\").\n"
        "6. Do NOT invent column names that are not in the source header list.\n"
        "7. If this file is not demand/inventory time-series data, reject with clear "
        "instructions.\n\n"
        f"Source headers: {json.dumps(cabeceras, ensure_ascii=False)}\n\n"
        "First rows (CSV):\n"
        f"{muestra.to_csv(index=False)}"
    )
