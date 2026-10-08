# Evaluacion 3.2 — Adaptador del conjunto de candidatos de M2

## Que se hizo

Implementacion del **handoff M2 -> M3** (tareas `TASK-EVAL-3.2-A1` y
`TASK-EVAL-3.2-A2`), segun ADR-03-004 y ADR-03-005.

1. **Esquema compartido del manifiesto**
   (`comun/modelos/manifiesto_candidatos.py`): una sola definicion para M2 y
   M3. Cada familia tiene su configuracion con todos los hiperparametros
   obligatorios.
2. **Productor de M2** (`optimizacion/router/manifiesto.py`):
   `construir_manifiesto` emite el ganador del HPO de cada familia para cada
   SKU. `pred-engine run` lo escribe como `candidatos.json` en la carpeta de
   la corrida.
3. **Validacion en dos niveles** (`adaptador_candidatos/validacion.py`,
   ADR-03-005): rechazo del lote completo (fail-closed) o fallo aislado de un
   candidato, siempre con su registro.
4. **Registro e instanciacion** (`adaptador_candidatos/registro.py`): las
   mismas fabricas que uso M2, y la linea base Seasonal Naive (m = 7).
5. **`pronosticar(observadas, H)` en todos los modelos** (`forecasting/base.py`
   y cada envoltorio, ADR-03-004): se ajusta una vez con la historia <= t\* y se
   pronostica desde cada origen de la reserva sin reestimar.

## Como

```
M2  SelectionResult (forecast_config, forecast_seed) por SKU y familia
      │ construir_manifiesto(run_id_m2, contexto de 3.1, emitido_en)
      ▼
    candidatos.json  (sobre tipado + candidatos crudos)
      │ validar_manifiesto(esperado=ContextoParticion, skus_panel, ...)
      ▼
    LoteRechazadoError(motivo)        ── la reserva no se abre (3.5: RECHAZADA)
    HandoffValidado(candidatos, fallos)
      │ instanciar(candidato)              instanciar_linea_base()
      ▼
    adaptador.fit(historia <= t*)  ──►  adaptador.pronosticar(observadas, H)
                                        por cada origen de la reserva (3.3)
```

### Rechazo del lote (fail-closed)

Un error que compromete la comparacion de **todos** los candidatos rechaza el
lote antes de abrir la reserva:

| Motivo | Cuando |
| --- | --- |
| `manifiesto_ilegible` | JSON invalido, sobre mal formado o candidato sin `candidato_id` o `sku` legibles |
| `version_no_soportada` | `schema_version` distinta de 1 |
| `contexto_incompatible` | la particion declarada (huella del Parquet de M1, t\*, fraccion) difiere de la de 3.1 |
| `handoff_posterior_a_reserva` | `emitido_en` no es anterior a la apertura de la reserva |
| `candidato_id_duplicado` | dos candidatos con el mismo id |
| `sku_fuera_del_panel` | un SKU que no esta en el panel de M1 |
| `error_no_clasificado` | cualquier otra excepcion: nunca degrada a un fallo local |

### Fallo de un candidato

Un error que solo afecta a un candidato produce un `FalloCandidato` (SKU,
candidato, modelo, campo, codigo y mensaje) con su `ERROR` en el log; los demas
siguen:

| Codigo | Cuando |
| --- | --- |
| `modelo_no_registrado` | familia o modelo desconocidos, o familia sin fabrica (modelo deshabilitado) |
| `campo_faltante` / `semilla_ausente` | falta un hiperparametro / falta la semilla |
| `campo_sobrante` | un campo que el esquema no admite |
| `tipo_incorrecto` | por ejemplo `2.0` o `true` donde va un entero |
| `fuera_de_dominio` | fuera de rango o de una regla del esquema |
| `pesos_no_disponibles` | no cargan los pesos de Chronos-2 (una sola carga por corrida) |

### `pronosticar` por familia (sin reestimar)

| Modelo | Como usa lo observado hasta el origen |
| --- | --- |
| SARIMA | `extend` filtra las observaciones nuevas con los parametros ya estimados |
| LightGBM | los lags y la posicion incluyen lo observado; los arboles no cambian |
| MLP | lo observado se normaliza con la media y escala del ajuste; los pesos no cambian |
| Chronos-2 | el contexto suma lo observado (hasta `max_contexto`); los pesos son fijos |
| Seasonal Naive | el ultimo ciclo de 7 dias incluye lo observado |

Sin observaciones nuevas, `pronosticar(vacio, H)` es igual a `predict(H)`.

## Donde

| Pieza | Ruta |
| --- | --- |
| Esquema compartido | `src/pred_engine/comun/modelos/manifiesto_candidatos.py` |
| Productor de M2 | `src/pred_engine/optimizacion/router/manifiesto.py` |
| Validacion, registro y contratos | `src/pred_engine/forecasting/adaptador_candidatos/` |
| `pronosticar` | `src/pred_engine/forecasting/base.py` y los envoltorios de `comun/modelos/` |
| Pruebas | `tests/comun/modelos/test_manifiesto_candidatos.py`, `tests/optimizacion/router/test_manifiesto.py`, `tests/forecasting/adaptador_candidatos/` |
| Constructores de prueba | `tests/_manifiestos.py` |
| API | `API_SPECIFICATION.md` |
| Notion | ADR-03-004, ADR-03-005; TASK-EVAL-3.2-A1/A2 |

## Mejoras respecto al texto de Notion (y por que)

- **Una sola definicion del manifiesto en `comun`.** ADR-03-004 pide un
  esquema comun, y `comun` no puede importar `optimizacion`. M2 lo produce y
  M3 lo valida con el mismo modelo Pydantic.
- **Configuraciones completas, sin defaults.** Un hiperparametro ausente nunca
  se completa con el default de la libreria: M3 evaluaria otra configuracion
  sin notarlo. Los rangos son los que aceptan los constructores, no los del
  espacio de HPO.
- **Chronos-2 declara su configuracion congelada** (`CHRONOS2_ZERO_SHOT`),
  aunque no se optimice: asi M3 verifica la revision de pesos y el cuantil
  puntual (0.5).
- **Los candidatos viajan crudos dentro del sobre** y se validan uno a uno: un
  candidato mal formado no rechaza el lote.
- **`ContextoParticion` usa los nombres de 3.0-A1** (`ingesta_ref_m1`,
  `t_corte_reserva`, `fraccion_reserva`) y llega como parametro: 3.2 no calcula
  la particion, la compara con la de 3.1.
- **El registro importa las fabricas directamente**, no desde
  `pipeline_setup`, para que el pipeline pueda depender de M3 sin ciclos. Son
  las mismas que uso M2, asi M3 construye exactamente lo que M2 evaluo.
- **3.2 no persiste ni reanuda.** El rechazo del lote es una excepcion; 3.5 lo
  guarda como `RECHAZADA` y persiste los fallos de candidato.
- **Pasa a M3 solo el ganador del HPO de cada familia**, no los demas trials:
  M2 no elige entre familias, eso lo hace M3 (3.4).

## Verificacion

```bash
uv run pytest tests/forecasting/adaptador_candidatos tests/comun/modelos/test_manifiesto_candidatos.py tests/optimizacion/router/test_manifiesto.py -q
uv run pytest tests/forecasting/adaptador_candidatos/test_ida_y_vuelta.py -m slow -q
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

- La prueba de ida y vuelta (`slow`) corre las estrategias reales de M2,
  construye el manifiesto, lo valida sin fallos e instancia cada candidato.
- `pyproject.toml`: pyright incluye `forecasting/adaptador_candidatos`.

## Fuera de alcance y pendientes

- Ajustar y pronosticar en cada ventana de la reserva (3.3, Derek) y orquestar
  M3 (3.0, Derek).
- Persistir el handoff y sus fallos (3.5).
- Ajustes internos de los envoltorios que no viajan en el manifiesto (por
  ejemplo `max_iter` de SARIMA o los hilos de LightGBM): quedan fijados porque
  M3 usa la misma fabrica que M2 y la version de codigo es parte de la
  identidad de la corrida.
- Chronos-2 en despliegue: usar `HF_HUB_OFFLINE=1` para que `from_pretrained`
  falle en vez de descargar pesos.
- Las unidades que fallan en M2 no viajan en el manifiesto (ver pendientes de
  3.4).
