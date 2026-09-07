# ADR-003: Motor de topologia Syntetos-Boylan desacoplado

**Date:** 2026-09-06
**Status:** Proposed
**Notion Task:** TASK-DATA-1.3-A1 / B1 / B2 / C1 / C2

## Context

El Modulo 2 enruta familias de modelos segun `sku_class`. Notion describe
la matriz ADI/CV² en espanol e "Irregular". El panel 1.2 ya es diario con
ceros estructurales. El CSV de proyecto trae columnas ERP extra.

## Decision

1. Literales de contrato: `smooth | intermittent | erratic | lumpy`.
2. Umbrales 1.32 / 0.49 con empate al cuadrante alto (`>=`).
3. ADI y CV² son funciones numpy puras; el group-by es `classify_panel`.
4. El Parquet solo gana `sku_class`. ADI/CV² no se densifican por fila.
5. `pred-engine classify` no usa LLM.
6. `select_canonical_columns` proyecta extras ERP sin `rename`.

## Rationale

- El router 2.2 y la literatura coinciden en Lumpy, no en Irregular.
- Clasificar antes del remuestreo destruiria la intermitencia.
- 1.4 y el jurado pueden auditar metricas via CLI/`TopologyArtifact`
  sin acoplar el Parquet a floats redundantes.

## Consequences

- Documentar sinonimos ES/EN en la API.
- `ingest` de CSVs con extras deja de fallar en la barrera.
- Sincronizar este ADR en Notion cuando se apruebe.

## Alternatives Considered

- **Etiquetas en espanol en el Parquet:** rechazado (rompe el Modulo 2).
- **Persistir adi/cv2 por fila:** rechazado (duplicacion, contrato mas ancho).
- **Clasificar el CSV crudo sin remuestreo:** rechazado (ADI=1.0 falso).
