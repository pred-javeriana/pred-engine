# Fase 0 — Simulacion de pre-ingesta (secciones 0.3 y 0.4)

## Que se hizo en esta sesion

La Fase 0 genera el **Ground Truth** sintetico (50 000+ registros) con el que se
somete a estres al framework PRED. Parte del Moving Block Bootstrap (`pred_engine.aumentacion.mbb`) y agrega la compuerta de validacion y
el handoff hacia el Modulo 1.

### 0.3 — Preservacion de restricciones fisicas y topologicas

| Tarea | Pieza | Que hace |
| --- | --- | --- |
| 0.3-A1 | `restricciones.rectificar_demanda_no_negativa` | Clipping asimetrico: eleva a 0 la demanda negativa, no toca los positivos, no muta la entrada, reporta conteo y %. |
| 0.3-A2 | `restricciones.truncar_a_unidades_enteras` | Truncamiento (no redondeo) al entero absoluto; columna tipada `int64`. |
| 0.3-A3 | `restricciones.limites_lead_time_desde_semilla` + `acotar_lead_time` | Deriva umbrales min/max de la semilla (nunca `< 1`), acota y emite advertencia trazable. |
| 0.3-B1 | `divergencia.evaluar_divergencia_parametrica` | Divergencia relativa de media y varianza vs semilla; tolerancia configurable (5 % por defecto); devuelve veredicto + estadisticos. |
| 0.3-B2 | `rechazo.generar_series_aceptadas` | Bucle de rechazo: descarta y re-remuestrea las series que rompen el umbral; `max_reintentos` → `DivergenceRejectionExhausted`; reproducible con semilla fija; registra la tasa de rechazo. |
| 0.3-B3 | `tests/aumentacion/` | Suite determinista de la compuerta (una prueba por restriccion + reproducibilidad). |

`restricciones.aplicar_restricciones_fisicas` compone A1 + A2 + A3 sobre el panel.

### 0.4 — Artefacto de salida y contrato de transicion (handoff al Modulo 1)

| Tarea | Pieza | Que hace |
| --- | --- | --- |
| 0.4-A1 | `contrato` | Contrato versionado (`CONTRACT_VERSION`) de las 4 columnas `sku_id, timestamp, demand_qty, lead_time_days`, orden canonico, `timestamp` ISO 8601. |
| 0.4-A2 | `conformidad.validar_conformidad_o_fallar` | Verifica presencia, orden y tipo de columnas + leyes de valor; `SchemaConformanceError` (critico) ante cualquier desviacion; genera reporte en toda ejecucion. |
| 0.4-B1 | `worm.escribir_una_sola_vez` | Guarda WORM: resuelve `raw/` desde `PRED_DATA_ROOT`, aborta si el destino existe, rechaza rutas absolutas y nombres con directorios. |
| 0.4-B2 | `exportador_csv.exportar_artefacto_csv` | Valida → serializa a CSV en orden canonico → delega en la guarda WORM → registra hash SHA-256 y conteo de filas (exige 50 000+). |
| 0.4-C1 | `fase0.ejecutar_fase_0` / `main` | Orquestador One-Shot: semilla → remuestreo (MBB + rechazo) → compuerta → conformidad → exportador. **No importa `pred_engine.ingesta` ni `pred_engine.forecasting`.** |
| 0.4-C2 | `bitacora` | Bitacora estructurada JSON (semilla, tasa de rechazo, rectificaciones, hash); se persiste en `{data_root}/logs/`, nunca en `raw/`. |

## Como

```bash
uv run python -m pred_engine.aumentacion.fase0 semilla_kaggle.csv \
  --data-root data \
  --period 7 --n-series 40 --seed 42
```

La semilla debe traer ya las 4 columnas canonicas. Cada corrida con la misma
semilla produce un artefacto con el mismo hash. El CSV se deposita en
`data/raw/panel_sintetico_fase0.csv` (inmutable: un segundo intento falla con
`WormOverwriteError`).

## Donde

| Pieza | Ruta |
| --- | --- |
| MBB | `src/pred_engine/aumentacion/mbb.py` |
| Compuerta 0.3 | `src/pred_engine/aumentacion/{restricciones,divergencia,rechazo}.py` |
| Contrato 0.4 | `src/pred_engine/aumentacion/{contrato,conformidad}.py` |
| Persistencia WORM 0.4 | `src/pred_engine/aumentacion/{worm,exportador_csv,rutas}.py` |
| Orquestacion 0.4 | `src/pred_engine/aumentacion/{fase0,bitacora}.py` |
| API | `API_SPECIFICATION.md` (este directorio) |
| Pruebas | `tests/aumentacion/` |

## Fuera de alcance

- Clasificacion topologica ADI/CV² → responsabilidad del Modulo 1.
- Lectura de semillas con cabeceras caoticas → la Fase 0 asume columnas
  canonicas; la alineacion semantica es 1.2.
- El `aumentar()` de `mbb.py` fija `np.random.default_rng(42)` internamente; el
  bucle de rechazo compone directamente `decompose_series` +
  `moving_block_bootstrap(rng=...)` + `compose_series` para controlar la semilla.
