# ADR-007: Preservacion del panel diario a traves de 1.3

**Date:** 2026-09-06
**Status:** Proposed
**Notion ADR:** ADR-01-012
**Notion Task:** TASK-DATA-1.4-A2

## Context

El clasificador debe enriquecer, no reescribir. Un group-by mal hecho
podria reordenar filas, rellenar demanda o perder SKU. Eso corromperia
las series que el Modulo 2 interpreta como historia real.

## Decision

1. Capturar el panel diario *antes* de `classify_daily_panel`.
2. `require_panel_preserved` compara identidad posicional de
   `sku_id`, `timestamp`, `demand_qty`, `lead_time_days`.
3. Solo se admite anadir `sku_class`.
4. La comprobacion corre antes de cualquier escritura en `processed/`.
5. No se recalculan ADI ni CV² en esta frontera.

## Rationale

La prueba de preservacion es un contrato de integracion, no una
reimplementacion del motor SBC. Si 1.3 cambia su implementacion y muta
demanda, 1.4 falla en vez de publicar el dano.

## Consequences

- `publish_classified_panel` exige `daily_panel`.
- Un reorden silencioso es un error tipado (`PanelPreservationError`).

## Alternatives Considered

- **Comparar solo el set (sku, timestamp) ignorando orden:** rechazado
  (A2 pide detectar reordenamientos).
- **Confiar en 1.3 sin asercion:** rechazado (A2 es la frontera).
