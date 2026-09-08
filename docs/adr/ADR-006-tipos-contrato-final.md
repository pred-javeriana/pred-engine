# ADR-006: Tipos y vocabulario del artefacto final

**Date:** 2026-09-06
**Status:** Proposed
**Notion ADR:** ADR-01-011
**Notion Task:** TASK-DATA-1.4-A1 / TASK-DATA-1.4-C1

## Context

Notion pide String, Datetime64, Float e Integer, mas un conjunto cerrado
de etiquetas. Pandas admite varios dtypes equivalentes (object, string,
int64, Int64) que no deben filtrarse al Modulo 2.

## Decision

1. `PANEL_DTYPES` es el origen de verdad de normalizacion:
   string / datetime64[ns] / float64 / int64 / string.
2. `sku_class ∈ {smooth, intermittent, erratic, lumpy}` (literales EN
   del ADR-003, no sinonimos Notion en espanol).
3. `sku_class` es constante por `sku_id`.
4. `lead_time_days >= 1` se hereda de 1.2; 1.4 no relaja el dominio.

## Rationale

Tipos pandas explicitos hacen reproducible el roundtrip pyarrow.
Etiquetas EN mantienen el contrato con el Selection Router.

## Consequences

- Una etiqueta `intermitente` rechaza el artefacto.
- `verify` relee el Parquet y vuelve a imponer el mismo contrato.

## Alternatives Considered

- **Categorical pandas para sku_class:** rechazado (acopla al consumidor).
- **Permitir demand_qty int64:** rechazado (el contrato pide Float).
