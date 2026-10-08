# Evaluacion 3.4 — Evaluacion retrospectiva

## Que se hizo en esta sesion

Implementacion de la **evaluacion sobre la reserva** y la **seleccion
definitiva** de M3 (tareas `TASK-EVAL-3.4-A1` y `TASK-EVAL-3.4-A2`, subtareas
A1.1, A1.2, A2.1 y A2.2 en Notion), segun ADR-03-003, ADR-03-006, ADR-03-007 y
ADR-03-008.

1. **Contratos de evaluacion** (`calculo_errores/contratos.py`): entrada minima
   `PronosticoFechado` -> `SerieCandidato` -> `EntradaSku`, y salida tipada
   `Metricas` / `MetricasCandidato` / `EvaluacionSku`.
2. **Lectura segura de la reserva** (`separar_reserva`): corta la serie del
   panel en t\* con el `ReserveCut` de 3.1 (`comun/reserva.py`).
3. **Metricas** (`calculo_errores/metricas.py`): MAE, RMSE, MASE, ME y la razon
   `r = RMSE_cand / RMSE_SN`, por ventana y agregadas.
4. **Seleccion por categoria** (`veredictos/seleccion.py`): mediana de `r` con
   compuerta frente a Seasonal Naive y desempate por simplicidad.
5. **Veredictos** (`veredictos/veredictos.py`): veredicto por SKU y resumen por
   categoria en un `ResultadoEvaluacion` consolidado.

## Como

```
M1 panel (sku) ──separar_reserva(serie, ReserveCut)──► historia <= t* , reserva real
3.3 (Derek)  ──► PronosticoFechado por origen, para la linea base y cada candidato
                    │
                    ▼  evaluar_sku(EntradaSku, ReserveCut)      (3.4-A1)
               EvaluacionSku: Metricas por ventana y agregadas, por candidato
                    │
                    ▼  emitir_veredictos([...], datos_sinteticos=...)   (3.4-A2)
               ResultadoEvaluacion
                 ├─ por categoria: familia campeona, medianas de r, conteos
                 └─ por SKU: veredicto, campeon, r, metricas campeon y SN, IQR
                            y todos los modelos evaluados con su r (incluida SN)
                    │
                    ▼  3.5 persiste  ──► M4 presenta
```

### Metricas (ADR-03-006), con e = y - y_hat

| Metrica | Definicion | No calculable |
| --- | --- | --- |
| MAE | `media(|e|)` | — |
| RMSE | `sqrt(media(e^2))` | — |
| ME | `media(e)`; positivo = sub-pronostico | — |
| MASE | `MAE / Q1`, `Q1 = media(|y_t - y_{t-7}|)` sobre la historia <= t\* | `historia_corta`, `escala_cero`, `escala_no_finita` |
| r | `RMSE_cand / RMSE_SN` en las ventanas validas para ambos | `rmse_linea_base_cero`, `sin_ventanas_comunes_con_linea_base` |

Una **ventana valida** tiene pronostico y observaciones reales finitos y de la
longitud de sus fechas. Las invalidas se cuentan (`n_ventanas_totales` frente
a `n_ventanas_validas`) y se excluyen. `n_pronosticos_validos` cuenta las
ventanas con pronostico valido aunque falten datos reales: asi se distingue
una falla del modelo de la falta de datos.

### Seleccion por categoria (ADR-03-007)

1. Familias elegibles: las que M2 entrego para esa `sku_class`.
2. Representante por familia y SKU: la configuracion de menor RMSE en la
   reserva.
3. Conjunto comun: SKUs con `r` calculable para todas las familias; los demas
   se excluyen con su causa.
4. Campeona: la familia con menor mediana de `r`.
5. Compuerta: si la menor mediana es >= 1, la campeona es Seasonal Naive y el
   resultado es adverso.
6. Empate (diferencia < `tolerancia_empate`): gana la mas simple segun
   `orden_simplicidad`, entre las familias que tambien tienen mediana < 1.
7. Sin SKUs comparables: la campeona es Seasonal Naive (`sin_skus_comparables`).

### Veredicto por SKU (ADR-03-008), primera regla que se cumple

| Veredicto | Condicion |
| --- | --- |
| `FALLO_TECNICO` | La familia campeona no tiene instancia con pronostico valido para el SKU, o todos sus candidatos fallaron |
| `NO_EVALUABLE` | Ninguna observacion valida en la reserva |
| `EVIDENCIA_INSUFICIENTE` | Menos de `n_min` (30) ventanas validas, o demanda real toda en cero |
| `VALIDADO` | Datos reales, ventanas suficientes, `r < 1` y sin `comparacion_incompleta` |
| `EXPLORATORIO` | Lo demas: datos sinteticos, `r >= 1` o no calculable, o `comparacion_incompleta` |

`VALIDADO` significa evidencia descriptiva suficiente y favorable, **no**
significancia estadistica. No se ejecutan pruebas Diebold-Mariano (ADR-03-008).

`n_min` es el minimo de **ventanas validas** (pronosticos emitidos desde un
origen dentro de la reserva y contrastados con lo observado) que necesita el
campeon de un SKU para que su evidencia cuente como suficiente.

### Que se ve por SKU

La familia campeona se elige por categoria, pero cada SKU usa su propia
instancia y su resultado queda visible: campeon, r, metricas del campeon y de
SN, veredicto y si pierde frente a SN. Ademas, `modelos_evaluados` lista como
dato descriptivo **todos** los modelos evaluados en el SKU con su r, incluida
Seasonal Naive (r = 1). Asi se ve, por ejemplo, que un SKU habria tenido mejor
r con SARIMA aunque la categoria la gane Chronos-2.

## Donde

| Pieza | Ruta |
| --- | --- |
| Metricas (A1) | `src/pred_engine/forecasting/evaluaciones/calculo_errores/` |
| Seleccion y veredictos (A2) | `src/pred_engine/forecasting/evaluaciones/veredictos/` |
| Pruebas | `tests/forecasting/evaluaciones/` |
| Constructores de prueba | `tests/_evaluaciones.py` |
| API | `API_SPECIFICATION.md` |
| Notion | ADR-03-003, ADR-03-006, ADR-03-007, ADR-03-008; TASK-EVAL-3.4-A1/A2 |

## Mejoras respecto al texto de Notion (y por que)

- **Entrada por ventanas, de cualquier cantidad.** Hoy el pipeline entrega un
  solo pronostico por candidato desde t\* para toda la reserva
  (`pronosticos.parquet`, ADR-017): es una ventana. Las ventanas de la reserva
  que piden ADR-03-003/004 son de 3.3 (Derek). 3.4 no depende de cuantas
  lleguen. Con una sola ventana, todo SKU queda en `EVIDENCIA_INSUFICIENTE`
  (`n_min = 30`), como corresponde.
- **"Apertura segura de la reserva"** es `separar_reserva` con el `ReserveCut`
  de `comun/reserva.py` (3.1 ya existe en codigo). Un pronostico con origen
  antes de t\* o con fechas dentro de la historia se rechaza con
  `EvaluacionRetrospectivaError`.
- **La linea base Seasonal Naive** llega como una serie mas
  (`familia = "seasonal_naive"`). Recorre las mismas ventanas que los
  candidatos (ADR-03-004); su propia `r` vale 1.
- **Familias elegibles = las entregadas por M2** para la categoria, no
  `FAMILIES_BY_SKU_CLASS`: `build_pipeline` excluye foundation por defecto, y
  usar la matriz completa dejaria vacio el conjunto comun.
- **Desempate acotado por la compuerta**: una familia con mediana >= 1 nunca
  gana por simplicidad.
- **El origen de los datos (real o sintetico)** llega como argumento de
  `emitir_veredictos` (`datos_sinteticos`), porque lo exige la regla de
  ADR-03-008, pero no se guarda en `ResultadoEvaluacion`: es un dato de la
  corrida y lo persiste 3.5. La tabla `ingestas` de la plataforma no lo guarda.
- **Todos los modelos evaluados por SKU con su r** (`modelos_evaluados`), como
  dato descriptivo: la seleccion definitiva sigue siendo por categoria.
- **Contratos internos como dataclasses congeladas**, no Pydantic: 3.4 no
  serializa; quien escribe es 3.5.

## Verificacion

```bash
uv run pytest tests/forecasting/evaluaciones -q
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest -m "not slow"
```

- `pyproject.toml`: `forecasting/evaluaciones/*` sale del `omit` de cobertura
  (el codigo nuevo se mide) y pyright incluye `calculo_errores` y `veredictos`.

## Fuera de alcance

- Generar las ventanas de la reserva (3.3, Derek) y orquestar M3 en el
  pipeline L4 (Derek).
- Persistir la evidencia (3.5).
- Pruebas Diebold-Mariano: descartadas por ADR-03-008; el stub
  `evaluaciones/diebold_mariano` no se toca.
