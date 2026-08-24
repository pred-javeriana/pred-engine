# Ingesta 1.2 — Alineacion semantica, validacion y continuidad temporal

## Que se hizo en esta sesion

1. **Sonda de cabeceras LLM** (`pred_engine.ingesta.sonda`): muestra de 5 filas,
   prompt zero-shot a temperatura 0.0, `DataFrame.rename` vectorizado, drop de
   columnas no mapeadas. Si el LLM no cubre el contrato, se aborta
   (`SemanticAlignmentError`): PRED no inventa columnas.
2. **Proveedores LLM modulares** (`pred_engine.comun.llm`): protocolo
   `LlmProvider` + adaptadores HTTP `gemini` / `openai` / `anthropic` via
   `httpx`, con timeout. La clave nunca se registra.
3. **Catalogo de modelos economicos** (`pred_engine.comun.llm.catalogo`):
   lista curada `AVAILABLE_MODELS` por proveedor, default al tier mas barato
   (`DEFAULT_MODELS`) y validacion fail-closed con `resolve_model` / `--model`.
4. **Contrato Pydantic v2** (`pred_engine.comun.modelos`) y **barrera fail-fast**
   (`pred_engine.ingesta.validador_formato`).
5. **Remuestreo diario** (`pred_engine.ingesta.continuidad`) con zero-filling
   de `demand_qty`.
6. **CLI** `pred-engine ingest` y `pred-engine models` para prueba de vida con
   `inventory_data.csv`.

## Como

La extraccion cruda sigue siendo 1.1 (`extract_csv`, todo `str`). 1.2 no escribe
en `raw/`. El baseline de mapeo del dataset de proyecto es:

| Canonico | Fuente |
| --- | --- |
| `timestamp` | `Date` |
| `sku_id` | `Item_ID` |
| `demand_qty` | `Avg_Usage_Per_Day` (nunca `Current_Stock`) |
| `lead_time_days` | `Restock_Lead_Time` |

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

Para pruebas reales del equipo con Gemini, usar `--model gemini-3.5-flash`
(incluido en el catalogo).

```bash
# Ver modelos permitidos
uv run pred-engine models --provider gemini

# Ingesta con modelo explicito
uv run pred-engine ingest \
  --csv inventory_data.csv \
  --provider gemini \
  --model gemini-3.5-flash \
  --api-key <GEMINI_KEY> \
  --data-root data
```

## Donde

| Pieza | Ruta |
| --- | --- |
| Contrato | `src/pred_engine/comun/modelos/` |
| LLM + catalogo | `src/pred_engine/comun/llm/` (`catalogo.py`, `fabrica.py`) |
| Sonda | `src/pred_engine/ingesta/sonda/` |
| Barrera | `src/pred_engine/ingesta/validador_formato/` |
| Remuestreo | `src/pred_engine/ingesta/continuidad/` |
| Pipeline / CLI | `src/pred_engine/ingesta/pipeline.py`, `src/pred_engine/cli.py` |
| API | `API_SPECIFICATION.md` (este directorio) |
| ADR | `docs/adr/ADR-002-proveedores-llm-y-alineacion-fail-closed.md` |

## Prueba de vida

```bash
uv run pred-engine ingest \
  --csv inventory_data.csv \
  --provider gemini \
  --model gemini-3.5-flash \
  --api-key <GEMINI_KEY> \
  --data-root data
```

## Fuera de alcance (sesiones futuras)

- Clasificacion topológica ADI/CV² (1.3, `ingesta/categorizacion`)
- UI de `pred-platform` (otro repositorio)
