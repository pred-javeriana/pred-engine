# ADR-005: Validador fail-closed del contrato de cinco columnas

**Date:** 2026-09-06
**Status:** Proposed
**Notion ADR:** ADR-01-010
**Notion Task:** TASK-DATA-1.4-A1

## Context

Un panel clasificado puede llegar con columnas extra, nulos o tipos
pandas ambiguos (object vs string). Publicarlo sin barrera romperia al
Modulo 2 de forma silenciosa.

## Decision

1. `validate_output_contract` exige exactamente `PANEL_FIELDS` en orden.
2. Fail-closed: un incumplimiento rechaza el panel completo.
3. La funcion no muta el DataFrame recibido.
4. `ClassifiedObservation` (Pydantic v2 strict) reutiliza el dominio de
   `InventoryObservation` y anade `sku_class`.

## Rationale

Misma filosofia que la barrera 1.2: estados invalidos no cruzan la
frontera. Reusar `InventoryObservation` evita duplicar `demand_qty >= 0`
y `lead_time_days >= 1`.

## Consequences

- El publicador normaliza dtypes en una copia *despues* de validar.
- Los logs no incluyen valores crudos de demanda (frontera publica).

## Alternatives Considered

- **Coercion silenciosa de tipos:** rechazado (oculta errores de 1.3).
- **Publicar filas validas y omitir invalidas:** rechazado (fail-closed).
