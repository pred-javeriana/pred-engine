# API — Ingesta 1.4 (contrato final y artefacto de salida)

Identificadores en ingles; comportamiento documentado para el jurado y
`pred-platform`. El Modulo 2 lee este contrato y no recalcula ADI/CV².

## Contrato persistido

Columnas exactas, en este orden:

| Columna | Tipo pandas | Dominio |
| --- | --- | --- |
| `sku_id` | String | no vacio |
| `timestamp` | Datetime64[ns] | sin nulos |
| `demand_qty` | Float64 | `>= 0`, sin nulos |
| `lead_time_days` | Int64 | `>= 1` (heredado de 1.2) |
| `sku_class` | String | `smooth` \| `intermittent` \| `erratic` \| `lumpy`, constante por `sku_id` |

Ruta: `{PRED_DATA_ROOT}/processed/<stem>.parquet`. Nunca CSV. Nunca `raw/`.

## `pred_engine.comun.modelos`

### `PANEL_FIELDS` / `PANEL_DTYPES`

`("sku_id", "timestamp", "demand_qty", "lead_time_days", "sku_class")`.
`PANEL_DTYPES` es el mapa de normalizacion antes de escribir.

### `ClassifiedObservation`

Pydantic v2, hereda `InventoryObservation` (`strict=True`, `extra=forbid`)
y anade `sku_class: SkuClass`. Una fila del artefacto final.

## `pred_engine.ingesta.salida`

### `require_positive_demand(panel) -> None`

Fail-closed. Cada `sku_id` debe tener al menos un `demand_qty > 0`.
Se ejecuta **antes** de `classify_daily_panel`. No calcula ADI/CV².

### `require_panel_preserved(daily_panel, classified) -> None`

Compara identidad posicional de las cuatro columnas transaccionales.
Permite solo `sku_class`. Detecta altas, bajas, duplicaciones, reorden
y mutaciones. No recalcula topologia.

### `validate_output_contract(frame) -> None`

Valida columnas exactas, tipos, nulos, dominios y constancia de
`sku_class`. **No muta** `frame`. Cualquier incumplimiento rechaza el
panel completo (`OutputContractError`).

### `normalize_output_frame(frame) -> DataFrame`

Copia con `PANEL_FIELDS` y `PANEL_DTYPES`. El origen queda intacto.

### `publish_classified_panel(classified, destination, *, data_root, daily_panel) -> Path`

Preserva → valida → normaliza → `export_parquet` en `processed/`.
Destino `.csv` o ruta bajo `raw/` se rechaza. No importa el paquete
`categorizacion`.

### `read_classified_parquet(path) -> DataFrame`

Lee con pyarrow y vuelve a imponer `validate_output_contract`. No escribe.

### Excepciones

`OutputHandoffError` (base, subclase de `ValueError`).
`OutputContractError`, `PanelPreservationError`, `HandoffPreconditionError`.

## `pred_engine.ingesta.categorizacion`

### `classify_daily_panel(frame) -> TopologyArtifact`

Alias de `classify_panel` (implementacion real de 1.3, no stub).
1.4 no reimplementa ADI, CV² ni el enrutador Syntetos-Boylan.

## `pred_engine.ingesta.pipeline`

`run_ingest`, `run_classify_csv` y `run_classify_parquet` publican solo
tras el handoff 1.4.

### `run_verify_parquet(parquet_path) -> DataFrame`

Relectura + contrato. Sin LLM y sin escritura.

## CLI

### `pred-engine classify`

Igual que 1.3 (sin LLM) mas validacion/publicacion 1.4.
Stdout incluye `contrato_1_4: accepted`.

### `pred-engine verify`

```
pred-engine verify --parquet PATH
```

Exit `0` ok, `6` contrato/preservacion/precondicion, `1` archivo ausente.

### `pred-engine ingest`

Camino con LLM. Ahora cierra en 1.4. Exit `6` si falla el handoff.
