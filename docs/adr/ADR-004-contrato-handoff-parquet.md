# ADR-004: Contrato de handoff y ruta de publicacion Parquet

**Date:** 2026-09-06
**Status:** Proposed
**Notion ADR:** ADR-01-009
**Notion Task:** TASK-DATA-1.4-B1 / TASK-DATA-1.4-C1

## Context

Los Modulos 2 y 3 deben consumir un artefacto inmutable, no el CSV crudo
ni un panel 1.2 sin `sku_class`. 1.3 ya escribia Parquet desde el
pipeline, pero la publicacion no era una frontera de contrato propia.

## Decision

1. El artefacto final vive solo en `{PRED_DATA_ROOT}/processed/*.parquet`.
2. Exactamente `PANEL_FIELDS`, nunca CSV del contrato final.
3. `publish_classified_panel` reutiliza `export_parquet` (guarda `raw/`)
   y no importa el motor SBC.

## Rationale

Parquet columnar + schema estable desacopla ingesta de forecasting.
Escribir desde 1.3 sin revalidar permitiria que un cambio del clasificador
publique un panel incompatible.

## Consequences

- `classify` e `ingest` solo persisten tras 1.4.
- MBB y el router 2.2 leen este Parquet, no `local_data/*.csv`.

## Alternatives Considered

- **CSV ademas de Parquet:** rechazado (B1 lo prohibe).
- **Persistir ADI/CV² por fila:** rechazado (ADR-003; 1.4 no lo reabre).
