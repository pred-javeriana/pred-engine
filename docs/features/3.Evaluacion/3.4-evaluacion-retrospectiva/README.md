# Evaluacion 3.4 — Evaluacion retrospectiva

## Que se hizo

Implementacion de la **evaluacion sobre la reserva** y la **seleccion
definitiva** de M3 (tareas `TASK-EVAL-3.4-A1` y `TASK-EVAL-3.4-A2`), segun
ADR-03-003, ADR-03-004, ADR-03-006, ADR-03-007 y ADR-03-008. Version de
metricas `3.4.1`; politica `3.4.1-inicial`.

1. **Contratos de evaluacion** (`calculo_errores/contratos.py`): entrada
   `PronosticoFechado` -> `SerieCandidato` -> `EntradaSku`, y salida tipada
   `Metricas` / `MetricasCandidato` / `EvaluacionSku`.
2. **Linea base** (`pronosticar_linea_base`): 3.4 genera Seasonal Naive (m = 7)
   en las mismas ventanas que los candidatos, con el adaptador de 3.2.
3. **Metricas** (`calculo_errores/metricas.py`): MAE, RMSE, MASE, ME y la razon
   `r = RMSE_cand / RMSE_SN`, por ventana y agregadas.
4. **Seleccion por categoria** (`veredictos/seleccion.py`): compiten las
   familias elegibles que declara la politica; mediana de `r` con compuerta
   frente a Seasonal Naive y desempate por simplicidad.
5. **Veredictos** (`veredictos/veredictos.py`): veredicto por SKU y resumen por
   categoria en un `ResultadoEvaluacion` consolidado.
6. **Diebold-Mariano** (`diebold_mariano/prueba.py`): prueba HLN-DM del
   campeon de cada SKU frente a Seasonal Naive, con correccion BH entre SKUs
   (ADR-03-008, alternativa 2). Acompana al veredicto sin cambiar sus reglas.

## Como

```
3.1  ReserveCut.of(panel) ──► t*, reserva (20 % final del calendario)
3.2  instanciar(candidato).fit(historia <= t*)
3.3 (Derek) ──► PronosticoFechado por origen, para cada candidato
                    │
                    ▼  evaluar_sku(EntradaSku, ReserveCut)              (3.4-A1)
               Seasonal Naive en las mismas ventanas (pronosticar_linea_base)
               EvaluacionSku: Metricas por ventana y agregadas, por modelo
                    │
                    ▼  emitir_veredictos([...], datos_sinteticos=...)   (3.4-A2)
               ResultadoEvaluacion
                 ├─ por categoria: familia campeona, medianas de r,
                 │                 familias excluidas, conteos
                 └─ por SKU: veredicto, campeon, r, metricas campeon y SN, IQR,
                            Diebold-Mariano y todos los modelos con su r
                    │
                    ▼  3.5 persiste  ──► M4 presenta
```

### Linea base (ADR-03-004)

`pronosticar_linea_base(entrada, corte)` trata a Seasonal Naive como un
adaptador mas:

- Toma las ventanas (origen y fechas) de los candidatos. Cada ventana debe ser
  los dias que siguen a su origen y ser igual entre candidatos; si no, lanza
  `EvaluacionRetrospectivaError`.
- Ajusta una vez con la historia <= t\* y pronostica cada origen con lo
  observado en la reserva hasta ese origen, sin reestimar.
- Si la historia tiene menos de 7 dias, o falta una observacion antes del
  origen, esa ventana queda sin valores finitos y cuenta como invalida.
- Su identificador es `{sku}/seasonal_naive/seasonal_naive`.

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

### Seleccion por categoria (ADR-03-003 y ADR-03-007)

1. Familias que compiten: las elegibles de la politica
   (`familias_elegibles`) que M2 entrego para esa `sku_class`. Las demas
   quedan en `familias_excluidas` con su causa: `no_entregada` (elegible, sin
   candidatos) o `no_elegible` (entregada fuera de la politica; su `r` sigue
   visible en `modelos_evaluados`).
2. Representante por familia y SKU: la configuracion de menor RMSE en la
   reserva.
3. Conjunto comun: SKUs con `r` calculable para todas las familias que
   compiten; los demas se excluyen con su causa.
4. Campeona: la familia con menor mediana de `r`.
5. Compuerta: si la menor mediana es >= 1, la campeona es Seasonal Naive y el
   resultado es adverso.
6. Empate (diferencia < `tolerancia_empate`): gana la mas simple segun
   `orden_simplicidad`, entre las familias que tambien tienen mediana < 1.
7. Sin SKUs comparables: la campeona es Seasonal Naive (`sin_skus_comparables`).

Familias elegibles de la politica inicial (la matriz 2.2 de enrutamiento de M2):

| Categoria | Familias |
| --- | --- |
| smooth, erratic | classical, ml, dl, foundation |
| intermittent | classical, ml, foundation |
| lumpy | classical, foundation |

### Veredicto por SKU (ADR-03-008), primera regla que se cumple

| Veredicto | Condicion |
| --- | --- |
| `FALLO_TECNICO` | La familia campeona no tiene instancia con pronostico valido para el SKU, o todos sus candidatos fallaron |
| `NO_EVALUABLE` | Ninguna observacion valida en la reserva |
| `EVIDENCIA_INSUFICIENTE` | Menos de `n_min` (30) ventanas validas, o demanda real toda en cero |
| `VALIDADO` | Datos reales, ventanas suficientes, `r < 1` y sin `comparacion_incompleta` |
| `EXPLORATORIO` | Lo demas: datos sinteticos, `r >= 1` o no calculable, o `comparacion_incompleta` |

`comparacion_incompleta` indica que algun candidato no llego a pronosticar
(`n_candidatos_fallidos`, en 3.2 o 3.3) o que un candidato de una familia
elegible no tiene ningun pronostico valido. Un candidato sin pronosticos de una
familia no elegible no cuenta.

`VALIDADO` significa evidencia descriptiva suficiente y favorable, **no**
significancia estadistica. `n_min` es el minimo de **ventanas validas**
(pronosticos emitidos desde un origen dentro de la reserva y contrastados con
lo observado) que necesita el campeon de un SKU.

### Diebold-Mariano (ADR-03-008, alternativa 2)

- **Que compara:** en cada SKU, la instancia de la familia campeona contra
  Seasonal Naive. No compara los demas candidatos ni las familias entre si.
- **Perdida:** el error cuadratico medio de cada ventana, coherente con `r`.
  El diferencial es `d_w = MSE_campeon - MSE_SN` en las ventanas validas para
  ambos.
- **Estadistico:** varianza de largo plazo con las autocovarianzas hasta el
  rezago `h - 1` (`h` = horizonte de las ventanas, que se solapan), factor
  HLN `sqrt((T + 1 - 2h + h(h - 1)/T) / T)`, t de Student con `T - 1` grados
  de libertad y p-valor bilateral, como `forecast::dm.test`. Un estadistico
  negativo indica menor perdida del campeon.
- **Multiplicidad:** Benjamini-Hochberg sobre todos los SKUs probados en la
  corrida, con nivel `alfa_dm` (0.05) de la politica.
- **Cuando no se prueba** queda la causa: `datos_sinteticos`,
  `campeon_es_linea_base`, `veredicto:<X>` (solo se prueban SKUs `VALIDADO` o
  `EXPLORATORIO`), `pocas_ventanas` (`T <= h`), `varianza_no_positiva` o
  `sin_ventanas_comunes`.

La prueba **no cambia el veredicto**: se adjunta en `VeredictoSku.diebold_mariano`
como evidencia aparte, y 3.5 la persiste para M4.

### Que se ve por SKU

La familia campeona se elige por categoria, pero cada SKU usa su propia
instancia y su resultado queda visible: campeon, r, metricas del campeon y de
SN, veredicto, prueba Diebold-Mariano y si pierde frente a SN. Ademas,
`modelos_evaluados` lista como dato descriptivo **todos** los modelos evaluados
en el SKU con su r, incluida Seasonal Naive (r = 1). Asi se ve, por ejemplo,
que un SKU habria tenido mejor r con SARIMA aunque la categoria la gane
Chronos-2.

## Donde

| Pieza | Ruta |
| --- | --- |
| Metricas y linea base (A1) | `src/pred_engine/forecasting/evaluaciones/calculo_errores/` |
| Seleccion y veredictos (A2) | `src/pred_engine/forecasting/evaluaciones/veredictos/` |
| Diebold-Mariano | `src/pred_engine/forecasting/evaluaciones/diebold_mariano/` |
| Pruebas | `tests/forecasting/evaluaciones/` |
| Constructores de prueba | `tests/_evaluaciones.py` |
| API | `API_SPECIFICATION.md` |
| Notion | ADR-03-003, ADR-03-004, ADR-03-006, ADR-03-007, ADR-03-008; TASK-EVAL-3.4-A1/A2 |

## Mejoras respecto al texto de Notion (y por que)

- **Entrada por ventanas, de cualquier cantidad.** Hoy el pipeline entrega un
  solo pronostico por candidato desde t\* para toda la reserva
  (`pronosticos.parquet`): es una ventana. Las ventanas de la reserva que piden
  ADR-03-003/004 son de 3.3 (Derek). 3.4 no depende de cuantas lleguen. Con
  una sola ventana, todo SKU queda en `EVIDENCIA_INSUFICIENTE` (`n_min = 30`).
- **3.4 no separa el 20 %.** El corte lo calcula `ReserveCut.of`
  (`comun/reserva.py`), el mismo que usan M1 y M2. 3.4 solo lee la reserva y
  rechaza con `EvaluacionRetrospectivaError` un pronostico con origen antes de
  t\* o con fechas dentro de la historia. `separar_reserva` solo aplica un
  corte ya calculado a una serie del panel.
- **La linea base la genera 3.4**, no llega en la entrada: asi recorre
  exactamente las ventanas de los candidatos, como pide ADR-03-004.
- **Familias elegibles predeclaradas en la politica** (ADR-03-003). Que M2
  entregue o no una familia se sabe antes de abrir la reserva, asi que
  excluirla con causa no es una decision post hoc.
- **Desempate acotado por la compuerta**: una familia con mediana >= 1 nunca
  gana por simplicidad.
- **El origen de los datos (real o sintetico)** llega como argumento de
  `emitir_veredictos` y queda en `ResultadoEvaluacion.datos_sinteticos`; 3.5
  lo contrasta con el de la corrida. La tabla `ingestas` de la plataforma no
  lo guarda.
- **Diebold-Mariano como evidencia, no como regla.** ADR-03-008 escogio
  veredictos descriptivos y dejo HLN-DM con correccion entre SKUs, solo sobre
  datos reales, para cuando se pida significancia. Se implementa esa
  alternativa sin tocar las reglas del veredicto. El ADR pide registrarla en
  un ADR nuevo.
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

- `pyproject.toml`: pyright incluye `calculo_errores`, `diebold_mariano` y
  `veredictos`.
- La distribucion t viene de `scipy`, que llega como dependencia obligatoria de
  `statsmodels`. Declararla aparte exige regenerar `uv.lock` con la version de
  uv de CI.

## Fuera de alcance y pendientes

- Generar las ventanas de la reserva (3.3) y orquestar M3 en el pipeline
  (3.0 y L4): Derek.
- Persistir la evidencia (3.5).
- Holm entre familias y Model Confidence Set (Estado del Arte, capa 5): ningun
  ADR los pide.
- Fallas de unidades de M2: no viajan en el manifiesto. Un SKU al que le falta
  una familia elegible por una falla de M2 sale del conjunto comun con causa,
  pero no queda con `comparacion_incompleta`. Decision pendiente.
