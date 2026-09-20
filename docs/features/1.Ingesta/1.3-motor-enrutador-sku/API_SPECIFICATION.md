# API — Ingesta 1.3 (motor de topologia SKU)

Identificadores en ingles; comportamiento documentado para el jurado y
`pred-platform`.

## Umbrales (literatura)

| Simbolo | Valor | Fuente |
| --- | --- | --- |
| ADI | 1.32 | Johnston & Boylan (1996); Syntetos, Boylan & Croston (2005) |
| CV² | 0.49 | Syntetos, Boylan & Croston (2005) |

Empates: `ADI >= 1.32` y `CV² >= 0.49` caen en el cuadrante **alto**.

## `pred_engine.comun.modelos`

### `ADI_THRESHOLD` / `CV2_THRESHOLD`

`1.32` y `0.49`. Unico origen de verdad; el enrutador no duplica literales.

### `SkuClass`

`Literal["smooth", "intermittent", "erratic", "lumpy"]`.

Sinonimos Notion (no se persisten):

| Contrato | Notion (ES) | Literatura |
| --- | --- | --- |
| `smooth` | Suave | Smooth |
| `intermittent` | Intermitente | Intermittent |
| `erratic` | Erratica | Erratic |
| `lumpy` | Irregular | Lumpy |

### `TOPOLOGY_FIELD` / `PANEL_FIELDS`

`sku_class`. `PANEL_FIELDS = CANONICAL_FIELDS + ("sku_class",)`.

Las columnas transaccionales (`sku_id`, `timestamp`, `demand_qty`,
`lead_time_days`) no se recodifican.

### `TopologyMetrics`

Pydantic v2, `strict=True`, `extra=forbid`.
Campos: `sku_id`, `n_periods` (>=1), `n_positive` (>=0), `adi` (>0),
`cv2` (>=0), `sku_class`. Una instancia por SKU, no por fila.

## `pred_engine.ingesta.categorizacion`

### `compute_adi(demand) -> float`

`ADI = n / n(d>0)` sobre observaciones finitas. NaN no es periodo.
Ceros y negativos no son demanda activa. `n(d>0)=0` → `TopologyMathError`.

### `compute_cv2(demand) -> float`

(σ/μ)² de valores estrictamente positivos. `ddof=1`. Un positivo → `0.0`.
Cero positivos → `TopologyMathError`.

### `route_syntetos_boylan(adi, cv2) -> SkuClass`

Matriz 2×2 con los umbrales anteriores.

### `select_canonical_columns(frame) -> DataFrame`

Copia con exactamente `CANONICAL_FIELDS`. No `rename`. Extras se omiten.

### `classify_panel(frame) -> TopologyArtifact`

Group-by `sku_id`. Una etiqueta por SKU, broadcast a todas las filas.
No muta el marco de entrada.

### `TopologyArtifact`

`frame` (columnas `PANEL_FIELDS`) + `metrics` (`tuple[TopologyMetrics, ...]`).
Solo `frame` se escribe a Parquet.

## `pred_engine.ingesta.pipeline`

### `run_ingest(...)`

Ahora: sonda → proyeccion → barrera → remuestreo → `classify_panel` → Parquet.
`IngestResult.topology` expone las metricas.

### `run_classify_csv(csv_path, *, data_root) -> (TopologyArtifact, Path)`

Sin LLM. Lee CSV (extras ERP permitidas), valida, remuestrea, clasifica.

### `run_classify_parquet(parquet_path, *, data_root) -> (TopologyArtifact, Path)`

Sin LLM. Proyecta columnas canonicas y clasifica el panel.

## CLI

### `pred-engine classify`

```
pred-engine classify --csv PATH [--data-root data]
pred-engine classify --parquet PATH [--data-root data]
```

Pase exactamente uno de `--csv` o `--parquet`. No requiere API key.

Stdout: tabla `sku_id,n_periods,n_positive,adi,cv2,sku_class` y
`sku_class_resumen`. Exit `0` ok, `3` esquema, `5` topologia, `1` resto.

### `pred-engine ingest`

Igual que 1.2, mas impresion del resumen de `sku_class`. Exit `5` si falla
la topologia. Columnas extra ya no rompen la barrera: se proyectan.
