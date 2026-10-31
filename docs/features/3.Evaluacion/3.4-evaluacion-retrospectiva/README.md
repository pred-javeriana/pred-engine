# Evaluacion 3.4 — Evaluacion retrospectiva

## Que se hizo

Implementacion de la **evaluacion sobre la reserva** y la **seleccion
definitiva** de M3 (tareas `TASK-EVAL-3.4-A1` y `TASK-EVAL-3.4-A2`), segun
ADR-03-003, ADR-03-004, ADR-03-006, ADR-03-007, ADR-03-008, ADR-03-011 y
ADR-023. Version de metricas `3.4.1`; politica `3.4.4-bloques`.

1. **Contratos de evaluacion** (`calculo_errores/contratos.py`): entrada
   `PronosticoFechado` -> `SerieCandidato` -> `EntradaSku`, y salida tipada
   `Metricas` / `MetricasCandidato` / `EvaluacionSku`.
2. **Linea base** (`pronosticar_linea_base`): 3.4 genera Seasonal Naive (m = 7)
   en las mismas ventanas que los candidatos, con el adaptador de 3.2.
3. **Metricas** (`calculo_errores/metricas.py`): MAE, RMSE, MASE, ME y la razon
   `r = RMSE_cand / RMSE_SN`, por ventana y agregadas.
4. **Bloques de la reserva** (`calculo_errores/bloques.py`): bloque de
   seleccion, embargo y bloque de prueba; `evaluar_bloques` da una vista de
   metricas por bloque y otra de toda la reserva.
5. **Seleccion por categoria** (`veredictos/seleccion.py`): compiten las
   familias elegibles que declara la politica; mediana de `r` con compuerta
   frente a Seasonal Naive y desempate por simplicidad, solo en el bloque de
   seleccion. El campeon queda congelado.
6. **Veredictos** (`veredictos/veredictos.py`): veredicto por SKU en el bloque
   de prueba y resumen por categoria en un `ResultadoEvaluacion` consolidado.
7. **Diebold-Mariano** (`diebold_mariano/prueba.py`): prueba HLN-DM del
   campeon de cada SKU frente a Seasonal Naive en el bloque de prueba, con BH
   dentro de cada categoria y el intervalo de `r` que invierte la prueba.
   Acompana al veredicto sin cambiar sus reglas.
8. **Confirmacion del campeon** (`confirmacion/`): el campeon congelado frente
   a cada familia competidora de su categoria, en el bloque de prueba.
9. **Seleccion de robustez** (`veredictos/seleccion_previa.py`): la misma
   regla de seleccion sobre los pliegues L3, previos a t\*.

## Como

```
3.1  fitting.reserve (L2) -> t*, reserva leidos de los metadatos del Parquet
3.2  instanciar(candidato).fit(historia <= t*)
3.3 ──► PronosticoFechado por origen, para cada candidato
                    │
                    ▼  evaluar_bloques(EntradaSku, ReserveCut, BloquesReserva)  (3.4-A1)
               Seasonal Naive en las mismas ventanas (pronosticar_linea_base)
               EvaluacionBloques: reserva, seleccion y prueba, cada una
               con Metricas por ventana y agregadas, por modelo
                    │
                    ▼  emitir_veredictos([...], bloques, datos_sinteticos=...,
                    │                    seleccion_previa=...)          (3.4-A2)
               ResultadoEvaluacion
                 ├─ por categoria: campeona del bloque de seleccion, medianas,
                 │                 conteos, significancia BH/BY, confirmacion,
                 │                 contendiente y seleccion de robustez
                 └─ por SKU (bloque de prueba): veredicto, campeon, r,
                            metricas campeon y SN, IQR, Diebold-Mariano,
                            intervalo de r y todos los modelos con su r
                    │
                    ▼  3.5 persiste  ──► M4 presenta
```

### Bloques de la reserva (ADR-023)

Con `R` dias reservados, horizonte `H` y `fraccion_seleccion` f de la politica
(0.4), `S = floor(f * R)`. El origen `t* + k` pronostica los dias `k + 1` a
`k + H`:

```
k = 0 .. S-H        bloque de seleccion  ventanas con todos sus dias en 1..S
k = S-H+1 .. S-1    embargo              cruzan el limite; nada las usa
k = S .. R-H        bloque de prueba     ventanas con todos sus dias en S+1..R
```

`BloquesReserva.de(corte, H, f)` fija los limites antes de abrir la reserva y
`limites()` es lo que guarda el protocolo de M3 (clave `bloques`).
`evaluar_bloques` calcula en una pasada tres vistas `EvaluacionSku`:
`reserva` (todas las ventanas), `seleccion` y `prueba`. Cada vista agrega solo
sus ventanas y cuenta las observaciones validas y los ceros de sus dias. Los
modelos se ajustan una sola vez en t\*; un origen del bloque de prueba usa las
observaciones hasta ese origen, tambien las del bloque de seleccion.

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

### Seleccion por categoria (ADR-03-003, ADR-03-007 y ADR-023)

La regla solo lee la vista `seleccion` de cada SKU:

1. Familias que compiten: las elegibles de la politica
   (`familias_elegibles`) que M2 entrego para esa `sku_class`. Las demas
   quedan en `familias_excluidas` con su causa: `no_entregada` (elegible, sin
   candidatos) o `no_elegible` (entregada fuera de la politica; su `r` sigue
   visible en `modelos_evaluados`).
2. Representante por familia y SKU: la configuracion de menor RMSE en el
   bloque de seleccion. Esa instancia representa a la familia en el SKU en todo
   lo que sigue, tambien en el bloque de prueba.
3. Conjunto comun: SKUs con `r` calculable y al menos `n_min` ventanas de
   seleccion comunes con Seasonal Naive para todas las familias que compiten.
   Los demas se excluyen con su causa (`pocas_ventanas:<familia>` cuando faltan
   ventanas).
4. Campeona: la familia con menor mediana de `r`.
5. Compuerta: si la menor mediana es >= 1, la campeona es Seasonal Naive y el
   resultado es adverso.
6. Empate (diferencia < `tolerancia_empate`): gana la mas simple segun
   `orden_simplicidad`, entre las familias que tambien tienen mediana < 1.
7. Sin SKUs comparables: la campeona es Seasonal Naive (`sin_skus_comparables`).

La familia campeona y las instancias quedan congeladas: ningun resultado del
bloque de prueba las cambia. `emitir_veredictos` rechaza bloques cuya fraccion
no es la de la politica.

Familias elegibles de la politica inicial (la matriz 2.2 de enrutamiento de M2):

| Categoria | Familias |
| --- | --- |
| smooth, erratic | classical, ml, dl, foundation |
| intermittent | classical, ml, foundation |
| lumpy | classical, foundation |

### Veredicto por SKU (ADR-03-008 y ADR-023), primera regla que se cumple

Cada condicion lee el bloque de prueba:

| Veredicto | Condicion en el bloque de prueba |
| --- | --- |
| `FALLO_TECNICO` | Ninguna familia del ranking de la categoria tiene instancia para el SKU, la instancia usada no tiene pronostico valido, o todos sus candidatos fallaron, tambien cuando ninguna familia entrego candidato en L2 |
| `NO_EVALUABLE` | Ninguna observacion valida en el bloque de prueba |
| `EVIDENCIA_INSUFICIENTE` | Menos de `n_min` (30) ventanas validas de prueba, o demanda real toda en cero |
| `VALIDADO` | Datos reales, ventanas suficientes, `r < 1` y sin `comparacion_incompleta` |
| `EXPLORATORIO` | Lo demas: datos sinteticos, `r >= 1` o no calculable, o `comparacion_incompleta` |

Un SKU con una sola familia evaluable al cerrar L2 no llega a 3.4, sea cual
sea el numero de familias enrutadas: sale con
`eligibility.fewer_than_two_families` (ADR-023) y la publicacion lo lista en
`m3_exclusiones`. Un SKU sin ninguna familia evaluable si llega y queda
`FALLO_TECNICO`.

Si la familia campeona no tiene instancia para un SKU, se usa la siguiente familia con instancia segun la mediana de r de la categoria en el bloque de seleccion (empates por nombre, nunca la mejor propia del SKU), y la justificacion lo registra.

`comparacion_incompleta` indica que algun candidato no llego a pronosticar
(`n_candidatos_fallidos`, en 3.2 o 3.3) o que un candidato de una familia
elegible no tiene ningun pronostico valido. Un candidato sin pronosticos de una
familia no elegible no cuenta.

`VALIDADO` significa evidencia descriptiva suficiente y favorable, **no**
significancia estadistica. `n_min` es el minimo de **ventanas validas**
(pronosticos emitidos desde un origen del bloque y contrastados con lo
observado) que necesita el campeon de un SKU en el bloque de prueba, y cada
familia de un SKU en el bloque de seleccion para entrar en el conjunto comun.

### Diebold-Mariano (ADR-03-011)

- **Que compara:** en cada SKU, la instancia de la familia campeona (o de la
  familia de fallback) del SKU contra Seasonal Naive, en el bloque de prueba.
  La comparacion entre familias es la confirmacion del campeon.
- **Perdida:** el error cuadratico medio de cada ventana, coherente con `r`.
  El diferencial es `d_w = MSE_campeon - MSE_SN` en las ventanas de prueba
  validas para ambos.
- **Estadistico:** varianza de largo plazo con las autocovarianzas hasta el
  rezago `h - 1` (`h` = horizonte de las ventanas, que se solapan), factor
  HLN `sqrt((T + 1 - 2h + h(h - 1)/T) / T)`, t de Student con `T - 1` grados
  de libertad y p-valor bilateral, como `forecast::dm.test`. Un estadistico
  negativo indica menor perdida del campeon.
- **Multiplicidad:** Benjamini-Hochberg dentro de cada categoria, sobre sus
  SKUs probados, con nivel `alfa_dm` (0.05) de la politica. El signo del
  estadistico da la direccion: mejor o peor que Seasonal Naive. Los mismos
  conteos con Benjamini-Yekutieli, valido bajo cualquier dependencia, se
  informan como sensibilidad (`SignificanciaCategoria`).
- **Intervalo de r:** el conjunto de `rho >= 0` para el que HLN-DM no rechaza
  sobre `L_campeon - rho^2 L_SN` al nivel `1 - alfa_dm` (Fieller). Contiene a
  `r` y excluye 1 justo cuando la prueba rechaza. Si la perdida de SN se
  distingue de cero, el conjunto es `[inferior, superior]`. Si no se distingue,
  no tiene cota superior: es `[inferior, infinito)`, con `superior` en `None`,
  o tiene dos tramos, `[inferior, superior]` y `[inferior_cola, infinito)`, y
  la prueba rechaza los valores entre ambos. Un SKU que BH selecciona lleva
  ademas el intervalo al nivel FCR `1 - R q / m` de su categoria, con `R`
  rechazos entre `m` pruebas (Benjamini y Yekutieli, 2005).
- **Cuando no se prueba** queda la causa: `datos_sinteticos`,
  `campeon_es_linea_base`, `veredicto:<X>` (solo se prueban SKUs `VALIDADO` o
  `EXPLORATORIO`), `pocas_ventanas` (`T <= h`), `varianza_no_positiva` o
  `sin_ventanas_comunes`.

La prueba **no cambia el veredicto**: se adjunta en `VeredictoSku.diebold_mariano`
como evidencia aparte, y 3.5 la persiste para M4. "Sin diferencia
significativa" nunca significa equivalencia.

Para combinar varias corridas, cada SKU conserva por corrida su p-valor
(`dm_p_valor`), la causa cuando no hay prueba valida (`dm_causa`) y el
estadistico (`dm_estadistico`), cuyo signo es el del diferencial medio de
perdida: negativo cuando el campeon pierde menos que Seasonal Naive.

### Grupo final de corridas (ADR-023)

La evidencia final es un grupo de tres corridas que solo difieren en la
semilla del motor: 0, 1 y 2. Cada corrida publica su significancia propia, con
BH dentro de cada categoria. Esa significancia por corrida no es la del grupo:
las tres corridas usan los mismos productos y dias, asi que no son evidencia
independiente. Quien agrupa las corridas calcula la significancia del grupo
con esta regla, y M3 sigue publicando por corrida:

1. Las entradas de cada corrida son `m3_veredictos.dm_p_valor`, el p crudo de
   HLN-DM, y el signo de `m3_veredictos.dm_estadistico`, que es el signo del
   diferencial de perdida campeon menos Seasonal Naive cuando la prueba es
   valida. Negativo significa que el campeon pierde menos.
2. Un producto entra en la familia de prueba del grupo solo si su
   `dm_p_valor` no es nulo en las tres corridas. Un producto sin p en alguna
   corrida queda fuera, por ejemplo si la regla de dos familias lo excluyo o si
   su prueba tiene `dm_causa`. El resumen del grupo lo lista con su estado en
   cada corrida.
3. El p del grupo es el mayor de sus tres p crudos. Es un p valido con
   cualquier dependencia entre corridas (interseccion-union, Berger 1982).
   Nunca se suman descubrimientos entre corridas, ni se promedian los p, ni se
   toma el mayor de los p disponibles, ni se combinan con Fisher o Stouffer.
4. BH se aplica una sola vez dentro de cada categoria, a q = `alfa_dm` (0.05),
   sobre los p del grupo de los productos que entran.
5. Un producto es significativamente mejor que Seasonal Naive en el grupo solo
   si BH lo rechaza y su estadistico es negativo en las tres corridas. Un
   producto con signos mixtos sigue en la familia de BH, porque cuenta para el
   umbral, pero nunca cuenta como mejor.

Ejemplo trabajado, con dos categorias. La columna de signos da el signo de
`dm_estadistico` en las semillas 0, 1 y 2.

| Categoria | Producto | p semilla 0 | p semilla 1 | p semilla 2 | Signos | p del grupo | p BH | Resultado |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| smooth | A | 0.001 | 0.004 | 0.002 | -, -, - | 0.004 | 0.012 | mejor |
| smooth | B | 0.010 | 0.030 | 0.020 | -, -, + | 0.030 | 0.045 | rechazado con signos mixtos |
| smooth | C | 0.020 | 0.010 | sin p | -, -, sin p | sin p | sin p | fuera de la familia |
| smooth | D | 0.200 | 0.400 | 0.300 | -, -, - | 0.400 | 0.400 | no rechazado |
| lumpy | E | 0.040 | 0.045 | 0.030 | -, -, - | 0.045 | 0.090 | no rechazado |
| lumpy | F | 0.300 | 0.010 | 0.020 | -, -, - | 0.300 | 0.300 | no rechazado |

- C no tiene p en la semilla 2, asi que la familia de smooth tiene m = 3. Si C
  entrara con el mayor de sus p disponibles (0.020), la familia tendria m = 4 y
  BH rechazaria tambien a C.
- B queda rechazado por BH (0.045 <= 0.05), pero su estadistico es positivo en
  la semilla 2: cuenta en la familia y no es mejor. Smooth tiene 1 producto
  mejor entre 3 probados.
- Lumpy es otra familia de BH. E tiene p < 0.05 en cada corrida, pero su p del
  grupo, 0.045, con m = 2 da p BH = 0.090. F tiene p = 0.010 en la semilla 1,
  pero su p del grupo es 0.300: una corrida sola no basta. Lumpy tiene 0
  productos mejores entre 2 probados. Un solo BH sobre las dos categorias daria
  otro umbral.

La verificacion de quien agrupa las corridas debe reproducir este ejemplo: el
producto sin p en una corrida, el producto con signos mixtos y la segunda
categoria.

### Confirmacion del campeon (ADR-023)

`confirmar_campeones` mide, en el bloque de prueba, si el campeon congelado de
cada categoria gana a cada familia competidora (las demas familias con mediana
en la seleccion). Para el campeon `f*` y la competidora `g`:

1. SKUs con ambas instancias y al menos `n_min_dm` ventanas de prueba comunes.
   Un SKU con perdida cero en esas ventanas tambien entra: un pronostico
   perfecto es evidencia valida.
2. `x_i = log(RMSE_f*,i / RMSE_g,i)` en esas ventanas. Si solo el campeon no
   tiene error, la razon vale 0 (`x = -inf`). Si solo la competidora no lo
   tiene, la razon es infinita (`x = +inf`). Si ninguno lo tiene, empatan con
   razon 1 (`x = 0`). No se suma ningun epsilon.
3. Wilcoxon de rangos con signo, bilateral (exacto con n <= 50). Los infinitos
   tienen los rangos mayores y los empates `x = 0` se descartan, como en
   `scipy`. Cuenta solo si la mediana de `x` es < 0; si no, su p vale 1
   (`p_iut`).
4. El p de la categoria es el mayor de sus competidoras (interseccion-union) y
   Holm lo ajusta entre las categorias probadas.
5. `confirmado` si el p de Holm es <= `alfa_dm` y, para cada competidora, el
   intervalo bootstrap del 95 % de la mediana de razones queda bajo 1. El
   bootstrap remuestrea bloques circulares de `bloque_bootstrap` (14) dias de
   prueba, `replicas_bootstrap` (2000) veces con `semilla_bootstrap`, y usa los
   mismos dias para todos los SKUs.

| Estado | Cuando |
| --- | --- |
| `confirmado` | Holm <= alfa y todos los intervalos bajo 1 |
| `no_confirmado` | Lo demas; la causa dice que fallo. El campeon no cambia |
| `sin_comparables` | Menos de `n_min_confirmacion` (10) SKUs para alguna competidora, o sin competidoras |
| `no_aplica` | El campeon es Seasonal Naive, o los datos no son reales declarados |

Cada comparacion informa tambien Hodges-Lehmann con su intervalo, la
proporcion de victorias del campeon con intervalo de Wilson y, por SKU, los
conteos de HLN-DM significativas con BH dentro de la comparacion. La
categoria lleva la mediana de `r` del campeon frente a Seasonal Naive con su
intervalo bootstrap.

**Perdidas cero.** Las medianas, los percentiles del bootstrap y Hodges-Lehmann
conservan las razones 0 e infinitas como bordes: un percentil entre un vecino
infinito y otro valor es ese infinito. En una replica del bootstrap entra cada
SKU con alguna de sus ventanas comunes entre los dias remuestreados. Un valor
queda sin definir cuando combina una razon 0 y una infinita: Hodges-Lehmann
con ambas, una mediana cuyos dos valores centrales son 0 e infinito, o un
intervalo bootstrap con una replica sin mediana o con un extremo entre una
replica de mediana 0 y otra de mediana infinita. Ese valor queda en `None` y la causa `perdida_cero:<campos>` lo nombra, por ejemplo
`perdida_cero:hodges_lehmann`, sin perder los conteos ni los demas efectos.
Una mediana sin definir no favorece al campeon, asi que su `p_iut` vale 1.

**Contendiente.** La misma regla de seleccion sin la familia fundacional elige
la mejor familia que no es Chronos-2 (`ResumenCategoria.contendiente`). Cuando
Chronos-2 es el campeon, su fila de comparacion frente a esa familia es la
comparacion "Chronos-2 contra la mejor familia no Chronos".

### Seleccion de robustez (ADR-023)

`evaluacion_previa` convierte los pliegues L3 de cada candidato, que solo usan
historia hasta t\*, en la misma vista que la regla de seleccion lee: cada
pliegue es una ventana cuyo origen es la fecha de su ultima observacion de
entrenamiento, y Seasonal Naive se ajusta con el entrenamiento de cada pliegue.
`seleccionar_previa` aplica la regla a cada categoria. M3 la calcula antes de
abrir la reserva y la guarda en la procedencia del protocolo: una reanudacion
lee la misma. Se informa en `ResumenCategoria.seleccion_previa` y nunca elige
al campeon, porque las familias optimizadas se puntuan en los pliegues que su
HPO uso.

### Que se ve por SKU

La familia campeona se elige por categoria, pero cada SKU usa su propia
instancia (o la de la siguiente familia del ranking si no tiene instancia de la
campeona) y su resultado en el bloque de prueba queda visible: campeon, r,
metricas del campeon y de SN, veredicto, prueba Diebold-Mariano con su
intervalo de r y si pierde frente a SN. Ademas,
`modelos_evaluados` lista como dato descriptivo **todos** los modelos evaluados
en el SKU con su r, incluida Seasonal Naive (r = 1). Asi se ve, por ejemplo,
que un SKU habria tenido mejor r con SARIMA aunque la categoria la gane
Chronos-2.

## Donde

| Pieza | Ruta |
| --- | --- |
| Metricas, linea base y bloques (A1) | `src/pred_engine/forecasting/evaluaciones/calculo_errores/` |
| Seleccion, veredictos y seleccion de robustez (A2) | `src/pred_engine/forecasting/evaluaciones/veredictos/` |
| Diebold-Mariano e intervalo de r | `src/pred_engine/forecasting/evaluaciones/diebold_mariano/` |
| Confirmacion del campeon | `src/pred_engine/forecasting/evaluaciones/confirmacion/` |
| Pruebas | `tests/forecasting/evaluaciones/` |
| Constructores de prueba | `tests/_evaluaciones.py` |
| API | `API_SPECIFICATION.md` |
| ADR | ADR-021, ADR-023 |
| Notion | ADR-03-003, ADR-03-004, ADR-03-006, ADR-03-007, ADR-03-008, ADR-03-011; TASK-EVAL-3.4-A1/A2 |

## Decisiones de implementacion

- **Entrada por ventanas, de cualquier cantidad.** El generador 3.3 entrega
  todas las ventanas diarias completas de H dias. La coordinacion 3.0 guarda
  sus pronosticos antes de evaluar. Con menos de 30 ventanas validas se
  conserva `EVIDENCIA_INSUFICIENTE`; nunca se acorta H para aumentar el conteo.
- **3.4 no separa el 20 %.** M3 recibe el corte de `fitting.reserve` de L2,
  que procede de los metadatos `pred_engine` de la publicacion Parquet.
  La importacion lo calcula una sola vez con `ReserveCut.of` sobre el calendario
  completo del CSV, antes de excluir productos. 3.4 solo lee esa reserva y
  rechaza con `EvaluacionRetrospectivaError` un pronostico con origen antes de
  t\* o con fechas dentro de la historia. `separar_reserva` solo aplica un
  corte ya calculado a una serie del panel.
- **La linea base la genera 3.4**, no llega en la entrada: asi recorre
  exactamente las ventanas de los candidatos, como pide ADR-03-004.
- **Familias elegibles predeclaradas en la politica** (ADR-03-003). Que M2
  entregue o no una familia se sabe antes de abrir la reserva, asi que
  excluirla con causa no es una decision post hoc.
- **Politica completa en el protocolo.** La version `3.4.4-bloques`, la
  fraccion de seleccion y los parametros de la confirmacion se guardan con el
  protocolo antes de abrir la reserva; otra politica es otra corrida.
- **La seleccion nunca recibe el bloque de prueba.** `seleccionar_categoria`
  solo ve la vista `seleccion`: cambiar cualquier observacion del bloque de
  prueba deja identico el registro de la seleccion.
- **Desempate acotado por la compuerta**: una familia con mediana >= 1 nunca
  gana por simplicidad.
- **El origen de los datos (real o sintetico)** llega como argumento de
  `emitir_veredictos` y queda en `ResultadoEvaluacion.datos_sinteticos`; 3.5
  lo contrasta con el de la corrida. La tabla `ingestas` de la plataforma no
  lo guarda.
- **Diebold-Mariano como evidencia, no como regla.** La decision
  documentada en ADR-03-011, ADR-021 y ADR-023 mantiene HLN-DM con BH
  activada por defecto, solo sobre datos reales, con BH dentro de cada
  categoria. La politica fija `n_min_dm` (30 por defecto) sobre origenes
  comunes. Si faltan, publica `pocas_ventanas`, no un p-valor. El horizonte es
  H y el resultado no cambia las reglas del veredicto.
- **Una confirmacion fallida no re-selecciona.** Elegir otra vez con el bloque
  de prueba volveria a mezclar seleccion y puntaje.
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

- 3.3 genera los pronosticos fechados de la reserva, 3.0/L4 coordina la
  evaluacion y su recuperacion, y 3.5 persiste y publica la evidencia. La
  presentacion visual de M4 no pertenece a esta capacidad.
- Model Confidence Set y Diebold-Mariano de panel (ADR-023, alternativas
  consideradas).
- El calculo del grupo final de corridas pertenece a quien las agrupa, con la
  regla de "Grupo final de corridas"; 3.4 publica por corrida sus entradas.
- Las fallas previas de M2 se conservan en la procedencia. M3 compara el
  handoff recibido y no inventa configuraciones que M2 no entrego. Las fallas
  de adaptacion o de cualquier ventana M3 marcan `comparacion_incompleta`.
