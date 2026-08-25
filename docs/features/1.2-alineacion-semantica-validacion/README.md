# Ingesta 1.2 — Alineacion semantica, validacion y continuidad temporal

## Que se hizo en esta sesion

1. **Sonda de cabeceras LLM consultiva** (`pred_engine.ingesta.sonda`): muestra de
   5 filas, prompt zero-shot a temperatura 0.0, payload JSON con
   `status: accepted|rejected` e instrucciones en `diagnostic[]`.
   **No hay `df.rename` ni `df.drop`**: el operador corrige el CSV manualmente.
2. **Proveedores LLM modulares** (`pred_engine.comun.llm`): protocolo
   `LlmProvider` + adaptadores HTTP `gemini` / `openai` / `anthropic` via
   `httpx`, con timeout. La clave nunca se registra.
3. **Catalogo de modelos economicos** (`pred_engine.comun.llm.catalogo`):
   lista curada `AVAILABLE_MODELS` por proveedor, default al tier mas barato
   (`DEFAULT_MODELS`) y validacion fail-closed con `resolve_model` / `--model`.
4. **Contrato Pydantic v2** (`HeaderDiagnostic`, `InventoryObservation`) y
   **barrera fail-fast** (`pred_engine.ingesta.validador_formato`).
5. **Remuestreo diario** (`pred_engine.ingesta.continuidad`) con zero-filling
   de `demand_qty`.
6. **CLI** `pred-engine probe` (solo diagnostico), `pred-engine ingest` y
   `pred-engine models`.

## Como

La extraccion cruda sigue siendo 1.1 (`extract_csv`, todo `str`). 1.2 no escribe
en `raw/`. La sonda actua como asistente consultivo:

| Resultado | Comportamiento |
| --- | --- |
| `rejected` | Imprime/registra JSON con instrucciones de renombrado → detiene el pipeline |
| `accepted` | El CSV ya tiene cabeceras canonicas exactas → continua a barrera Pydantic |

Un CSV estilo ERP (p. ej. `Date`, `Item_ID`, `Avg_Usage_Per_Day`) sera
**rechazado** con instrucciones como renombrar `Date` → `timestamp`. Tras
corregir el archivo, el operador vuelve a ejecutar `probe` o `ingest`.

### Seleccion de proveedor y modelo

El operador elige **proveedor** y **modelo** en el CLI. Solo se aceptan IDs
listados en `AVAILABLE_MODELS`; un modelo fuera del catalogo lanza
`UnknownModelError`. Si no se pasa `--model`, se usa el default economico del
proveedor:

| Proveedor | Default (mas barato) |
| --- | --- |
| `gemini` | `gemini-2.5-flash-lite` |
| `openai` | `gpt-4.1-nano` |
| `anthropic` | `claude-haiku-4-5` |

## Donde

| Pieza | Ruta |
| --- | --- |
| Contrato | `src/pred_engine/comun/modelos/` |
| LLM + catalogo | `src/pred_engine/comun/llm/` (`catalogo.py`, `fabrica.py`) |
| Sonda consultiva | `src/pred_engine/ingesta/sonda/` |
| Barrera | `src/pred_engine/ingesta/validador_formato/` |
| Remuestreo | `src/pred_engine/ingesta/continuidad/` |
| Pipeline / CLI | `src/pred_engine/ingesta/pipeline.py`, `src/pred_engine/cli.py` |
| API | `API_SPECIFICATION.md` (este directorio) |

## Prueba con archivo real

Diagnostico sin mutar (recomendado primero):

```bash
uv run pred-engine probe \
  --csv inventory_data.csv \
  --provider gemini \
  --model gemini-3.5-flash \
  --api-key <GEMINI_KEY> \
  --data-root data
```

Si la sonda rechaza, corrija el CSV segun el JSON de `diagnostic[]` y repita.
Cuando el CSV tenga cabeceras canonicas (`sku_id`, `timestamp`, `demand_qty`,
`lead_time_days`), `ingest` completara barrera + remuestreo + Parquet:

```bash
uv run pred-engine ingest \
  --csv inventory_data_canonico.csv \
  --provider gemini \
  --model gemini-3.5-flash \
  --api-key <GEMINI_KEY> \
  --data-root data
```

## Fuera de alcance (sesiones futuras)

- Clasificacion topologica ADI/CV² (1.3, `ingesta/categorizacion`)
- UI de `pred-platform` (otro repositorio)
