# Ingesta 1.1 — Extracción y almacenamiento crudo

## Qué se hizo en esta sesión

Se implementó el protocolo de ingesta pasiva del framework PRED:

1. **Logger JSON estructurado** (`pred_engine.comun.logger`) hacia stdout.
2. **Árbol inmutable** `raw/` / `staging/` / `processed/` bajo un `data_root` configurable.
3. **Guardia de E/S** que bloquea `open()` en modos de escritura bajo `raw/`.
4. **Extractor CSV pasivo** (sin inferir tipos) + hash SHA-256 del archivo en disco.
5. **Exportador Parquet** (motor PyArrow, compresión Snappy) hacia `processed/`.

## Cómo

- El logger usa `logging` de la biblioteca estándar y un `JsonFormatter` propio.
- El layout es una función pura respecto a memoria (solo toca el filesystem de forma idempotente).
- El extractor y el exportador no guardan DataFrames en variables de módulo.
- Las pruebas viven en `tests/comun/logger/` y `tests/ingesta/`.

## Dónde

| Pieza | Ruta |
| --- | --- |
| Formateador y logger | `src/pred_engine/comun/logger/` |
| Layout y guardia raw | `src/pred_engine/ingesta/data/` |
| CSV / Parquet | `src/pred_engine/ingesta/lector/` |
| API | `API_SPECIFICATION.md` (este directorio) |
| ADR de la raíz configurable | `docs/adr/ADR-001-raiz-de-datos-configurable.md` |

## Fuera de alcance (sesiones futuras)

- Clasificacion topológica ADI/CV² (TASK 1.3)
- UI de `pred-platform`

La sonda LLM, Pydantic y el remuestreo diario se implementaron en
`docs/features/1.2-alineacion-semantica-validacion/`.
