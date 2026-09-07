# Ingesta 1.3 — Motor de topologia SKU (Syntetos-Boylan)

## Que se hizo en esta sesion

1. **Contrato** `TopologyMetrics` + literales `sku_class` y umbrales 1.32 / 0.49.
2. **ADI** y **CV²** como funciones numpy 1-D, sin estado, con errores tipados.
3. **Enrutador** de la matriz Syntetos-Boylan (empates al cuadrante alto).
4. **`classify_panel`**: una etiqueta por SKU, columna `sku_class` en el panel.
5. **Proyeccion** de extras ERP a `CANONICAL_FIELDS` (el CSV real las trae).
6. **CLI** `pred-engine classify` (sin LLM) e inyeccion en `ingest`.

## Como

El panel diario de 1.2 (ceros estructurales) es el insumo correcto de ADI.
Clasificar el CSV crudo (solo filas con demanda > 0) daria ADI = 1.0 para
todos los SKUs del dataset de proyecto y mentiria al Modulo 2.

| ADI | CV² | `sku_class` |
| --- | --- | --- |
| < 1.32 | < 0.49 | `smooth` |
| >= 1.32 | < 0.49 | `intermittent` |
| < 1.32 | >= 0.49 | `erratic` |
| >= 1.32 | >= 0.49 | `lumpy` |

Las metricas ADI/CV² se imprimen en CLI y viven en `TopologyArtifact.metrics`.
El Parquet solo anade `sku_class` para no inflar el contrato hacia 1.4 / Modulo 2.

## Donde

| Pieza | Ruta |
| --- | --- |
| Contrato | `src/pred_engine/comun/modelos/contrato.py` |
| Nucleo | `src/pred_engine/ingesta/categorizacion/` |
| Pipeline / CLI | `pipeline.py`, `cli.py` |
| API | `API_SPECIFICATION.md` (este directorio) |
| ADR | `docs/adr/ADR-003-topologia-syntetos-boylan.md` |

## Prueba con archivo real (sin LLM)

```bash
uv run pred-engine classify \
  --csv local_data/inventory_data.csv \
  --data-root data
```

Esperado en este dataset: 10 SKUs, todos `intermittent` (ADI ~7–12 por el
zero-filling; CV² < 0.49). El Parquet queda en `data/processed/inventory_data.parquet`.

`ingest` sigue disponible si se quiere el camino con sonda LLM; ahora
tolera columnas extra.

## Fuera de alcance (sesiones futuras)

- Data augmentation / MBB (1.4, `ingesta` no; `aumentacion/`)
- Selection router del Modulo 2 (lee `sku_class`, no lo recalcula)
