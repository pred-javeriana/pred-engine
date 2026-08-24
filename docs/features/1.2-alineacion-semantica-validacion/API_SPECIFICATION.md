# API — Ingesta 1.2

Identificadores en ingles; comportamiento documentado para el jurado y
`pred-platform`.

## `pred_engine.comun.modelos`

### `CANONICAL_FIELDS`

Tupla `("sku_id", "timestamp", "demand_qty", "lead_time_days")`.

### `InventoryObservation`

Pydantic v2, `strict=True`, `extra=forbid`.
`demand_qty: float >= 0`, `lead_time_days: int >= 1`, `timestamp: datetime`,
`sku_id: str` no vacio.

### `HeaderMapping`

Canonico → nombre de columna fuente. `None` = no alineado.
`unmapped_fields()`, `source_to_canonical()`.

## `pred_engine.comun.llm`

### `LlmProvider.complete(prompt, *, temperature, timeout) -> str`

Contrato sin estado. La sonda fija `temperature=0.0`.

### `build_llm_provider(name, api_key, *, model=None) -> LlmProvider`

Nombres: `gemini` (alias `google`), `openai` (`gpt`), `anthropic` (`claude`).

### Catalogo de modelos (`pred_engine.comun.llm.catalogo`)

#### `AVAILABLE_MODELS`

Diccionario `proveedor → tuple[str, ...]` con IDs permitidos (tier economico,
ago 2026):

| Proveedor | Modelos permitidos |
| --- | --- |
| `gemini` | `gemini-2.5-flash-lite`, `gemini-2.0-flash-lite`, `gemini-2.5-flash`, `gemini-3.1-flash-lite`, `gemini-3-flash-preview`, `gemini-3.5-flash`, `gemini-2.0-flash` |
| `openai` | `gpt-4.1-nano`, `gpt-5.4-nano`, `gpt-5-mini`, `gpt-4.1-mini`, `gpt-4o-mini` |
| `anthropic` | `claude-haiku-4-5`, `claude-haiku-4-5-20251001` |

#### `DEFAULT_MODELS`

Default economico si `model=None`:

| Proveedor | Default |
| --- | --- |
| `gemini` | `gemini-2.5-flash-lite` |
| `openai` | `gpt-4.1-nano` |
| `anthropic` | `claude-haiku-4-5` |

#### `resolve_model(provider, model=None) -> str`

Valida que el modelo este en `AVAILABLE_MODELS`; si no, lanza
`UnknownModelError` con la lista de opciones.

#### `get_available_models(provider) -> tuple[str, ...]`

Devuelve la tupla del catalogo para un proveedor canonico.

### Excepciones

`LlmTimeoutError`, `LlmProviderError`, `UnknownProviderError`,
`UnknownModelError`.

## `pred_engine.ingesta.sonda`

### `probe_headers(frame, provider, *, timeout=30.0, n_rows=5) -> AlignmentArtifact`

Temperatura fijada a `0.0`. `AlignmentArtifact(frame, mapping, dropped_columns)`.

### `SemanticAlignmentError`

El dataset no es utilizable por PRED.

## `pred_engine.ingesta.validador_formato`

### `validate_aligned_frame(frame) -> DataFrame`

Fail-fast. Salida con `timestamp` `datetime64[ns]`, `demand_qty` float64,
`lead_time_days` int64, `sku_id` string.

### `SchemaBarrierError`

Incluye `row_index`, `column`, `raw_value` cuando aplica.

## `pred_engine.ingesta.continuidad`

### `resample_daily(frame) -> DataFrame`

Grid diario por `sku_id`, left join, `demand_qty` NaN → 0.

## `pred_engine.ingesta.pipeline`

### `run_ingest(csv_path, provider, *, data_root=None, timeout=30.0) -> IngestResult`

Deposita en `raw/`, extrae, alinea, valida, remuestrea, escribe Parquet en
`processed/`.

## CLI

### `pred-engine models --provider gemini|openai|anthropic`

Lista modelos permitidos y marca el default economico.

### `pred-engine ingest`

```
pred-engine ingest \
  --csv PATH \
  --provider gemini|openai|anthropic \
  [--api-key KEY] \
  [--model NAME] \
  [--data-root data] \
  [--timeout 30]
```

- `--model` debe estar en `AVAILABLE_MODELS` del proveedor elegido.
- Sin `--model`, se usa `DEFAULT_MODELS[provider]`.
- Clave: `--api-key` o `PRED_LLM_API_KEY` / `GEMINI_API_KEY` / `OPENAI_API_KEY` /
  `ANTHROPIC_API_KEY` (segun proveedor).

Exit codes: `0` ok, `2` alineacion, `3` esquema, `4` timeout LLM, `1` resto.
