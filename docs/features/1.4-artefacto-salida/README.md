# Ingesta 1.4 — Contrato de datos final y artefacto de salida

## Que se hizo en esta sesion

1. **Contrato Pydantic** `ClassifiedObservation` (extiende `InventoryObservation`
   con `sku_class`) y mapa `PANEL_DTYPES`.
2. **Validador fail-closed** de las cinco columnas del handoff. No muta el
   DataFrame. Rechaza el panel completo ante cualquier incumplimiento.
3. **Preservacion del panel diario:** 1.3 solo puede anadir `sku_class`.
   Altas, bajas, reorden o mutacion transaccional detienen la persistencia.
4. **Publicador Parquet** hacia `{PRED_DATA_ROOT}/processed/`. Sin CSV.
   Reutiliza `export_parquet` (1.1) para la guarda de `raw/`.
5. **Gate de demanda positiva** antes de clasificar, para no invocar ADI
   sobre series indefinidas.
6. **CLI** `pred-engine verify` y cableado de 1.4 en `classify` / `ingest`.

## Como

El pipeline queda:

```
CSV/Parquet → (sonda opcional) → barrera 1.2 → remuestreo diario
  → require_positive_demand
  → classify_daily_panel (1.3 real, no stub)
  → require_panel_preserved
  → validate_output_contract
  → Parquet en processed/
```

`sku_class` es la unica columna nueva. ADI/CV² no se persisten ni se
recalculan en 1.4: el publicador no importa `categorizacion`.

## Donde

| Pieza | Ruta |
| --- | --- |
| Contrato | `src/pred_engine/comun/modelos/contrato.py` |
| Handoff 1.4 | `src/pred_engine/ingesta/salida/` |
| Pipeline / CLI | `pipeline.py`, `cli.py` (`classify`, `ingest`, `verify`) |
| API | `API_SPECIFICATION.md` (este directorio) |
| ADR | `docs/adr/ADR-004` … `ADR-007` |

## Prueba con archivo real (sin LLM)

```bash
uv run pred-engine classify \
  --csv local_data/inventory_data.csv \
  --data-root data

uv run pred-engine verify \
  --parquet data/processed/inventory_data.parquet
```

Esperado en este dataset: 4862 filas, 10 SKU, todos `intermittent`.
Exit `0` ok, `6` si el contrato/preservacion/precondicion fallan.

## Mejoras respecto al texto de Notion (y por que)

- **`classify_daily_panel` es un alias de `classify_panel`.** Notion nombra
  `classify_daily_panel`; 1.3 ya existia como `classify_panel`. El alias
  evita romper tests y deja el nombre del contrato visible en 1.4.
- **Paquete `salida/` en vez de inflar `pipeline.py`.** El validador, la
  preservacion y el publicador son funciones puras reutilizables por
  cualquier vertical; el pipeline solo las compone.
- **Gate de demanda positiva antes de 1.3.** C1 pide fallar *antes* de
  clasificar. Depender de `TopologyMathError` (division por cero en ADI)
  mezclaria un error matematico de 1.3 con la frontera de publicacion.
- **`pred-engine verify`.** C1 exige releer el Parquet y revalidar tipos.
  Un subcomando de solo lectura evita repetir la clasificacion y sirve
  como auditoria del artefacto ya publicado.
- **Codigo de salida CLI 6** para errores 1.4, distinto de 3 (esquema 1.2)
  y 5 (topologia 1.3), para que el operador sepa en que frontera fallo.
- **Makefile `test-cov`.** El DoD de Notion cita `make test-cov` y el repo
  no tenia Makefile; el target delega a `uv run pytest` (umbral 80 % ya
  esta en `pyproject.toml`).
- **1.4 no es MBB.** El README de 1.3 etiquetaba augmentation como 1.4.
  El plan Notion de esta sesion es el contrato de salida; MBB sigue en
  `aumentacion/`.

## Fuera de alcance

- Data augmentation / MBB (`src/pred_engine/aumentacion/`)
- Selection router del Modulo 2
- Recalculo de ADI/CV² para enrutar (usar `sku_class` persistido)
