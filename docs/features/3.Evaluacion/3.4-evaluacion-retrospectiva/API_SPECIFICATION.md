# API — Evaluacion 3.4 (Evaluacion retrospectiva)

Identificadores en espanol; comentarios y mensajes en espanol. De los
candidatos 3.4 recibe arreglos y fechas: no los ajusta, no lee ni escribe
archivos. El unico modelo que ajusta es la linea base Seasonal Naive.

Puntos de entrada:

- Metricas, linea base y bloques (A1): `pred_engine.forecasting.evaluaciones.calculo_errores`
- Seleccion y veredictos (A2): `pred_engine.forecasting.evaluaciones.veredictos`
- Diebold-Mariano: `pred_engine.forecasting.evaluaciones.diebold_mariano`
- Confirmacion del campeon: `pred_engine.forecasting.evaluaciones.confirmacion`

## Constantes

| Nombre | Valor |
| --- | --- |
| `VERSION_METRICAS` | `"3.4.1"` |
| `PERIODO_ESCALA` | `7` (m de Q1, igual que Seasonal Naive) |
| `FAMILIA_LINEA_BASE` | `"seasonal_naive"` |
| `MODELO_LINEA_BASE` | `"seasonal_naive"` |
| `FAMILIA_FUNDACIONAL` | `"foundation"` (la familia que el contendiente deja fuera) |
| `ESTADOS_CONFIRMACION` | `confirmado`, `no_confirmado`, `sin_comparables`, `no_aplica` |
| `FAMILIAS_ELEGIBLES_INICIALES` | `sku_class -> familias`, la matriz 2.2 de M2 (ver README) |
| `POLITICA_INICIAL` | `PoliticaSeleccion()` (ver abajo) |
| `VEREDICTOS` | `FALLO_TECNICO`, `NO_EVALUABLE`, `EVIDENCIA_INSUFICIENTE`, `VALIDADO`, `EXPLORATORIO` |

## Entrada (`calculo_errores/contratos.py`)

Dataclasses congeladas (`eq=False` las que llevan arreglos).

### `PronosticoFechado`

| Campo | Tipo | Regla |
| --- | --- | --- |
| `origen` | `pd.Timestamp` | `>= t*` |
| `fechas` | `pd.DatetimeIndex` | los dias que siguen a `origen`, todos `> t*` |
| `valores` | `np.ndarray` | misma longitud que `fechas` para ser valido |

### `SerieCandidato`

`candidato_id: str`, `familia: str`, `modelo: str`,
`pronosticos: tuple[PronosticoFechado, ...]` (un origen no puede repetirse).

### `EntradaSku`

| Campo | Tipo |
| --- | --- |
| `sku` | `str` |
| `sku_class` | `SkuClass` |
| `historia` | `np.ndarray` (<= t\*) |
| `reserva` | `pd.Series` indexada por fecha (> t\*) |
| `candidatos` | `tuple[SerieCandidato, ...]` |
| `n_candidatos_fallidos` | `int` (default 0): candidatos que no llegaron a pronosticar |

La linea base no llega en la entrada: `evaluar_sku` la genera. Para la salida
actual del pipeline: un `PronosticoFechado(origen=t*,
fechas=reserve.reserved_dates, valores=FittedCandidate.forecast)` por candidato.

## Metricas (`calculo_errores/metricas.py`)

```python
separar_reserva(serie: pd.Series, corte: ReserveCut) -> tuple[np.ndarray, pd.Series]
escala_q1(historia: np.ndarray, m: int = 7) -> tuple[float | None, str | None]
pronosticar_linea_base(entrada, corte, *, ventanas_planeadas=None, confirmados=()) -> SerieCandidato
evaluar_sku(entrada, corte, *, linea_base=None) -> EvaluacionSku
evaluar_bloques(entrada, corte, bloques, *, linea_base=None) -> EvaluacionBloques
pares_comunes(candidato, linea_base) -> list[tuple[Metricas, Metricas]]
resumir_ventanas(serie, validas, base, escala, *, n_ventanas_totales, n_pronosticos_validos) -> MetricasCandidato
```

- `pronosticar_linea_base`: Seasonal Naive (`instanciar_linea_base()` de 3.2)
  ajustado una vez con `historia` y pronosticado en cada ventana de los
  candidatos con `pronosticar(observadas, H)`. Ventana sin valores finitos si
  no se puede ajustar o falta una observacion antes del origen.
- `evaluar_sku`: genera la linea base y calcula las metricas de ella y de cada
  candidato en toda la reserva.
- `evaluar_bloques`: lo mismo en una pasada, con las vistas `reserva`,
  `seleccion` y `prueba`. `EvaluacionRetrospectivaError("bloques de otra
  reserva")` si `bloques` no tiene el t\* y los dias de `corte`.
- `pares_comunes`: metricas de las ventanas validas para ambos, en el orden
  del candidato.
- `resumir_ventanas`: metricas por ventana y agregadas desde pares
  (real, pronostico) por origen; la usa la seleccion de robustez.

### `BloquesReserva` (`calculo_errores/bloques.py`, congelada)

| Miembro | Descripcion |
| --- | --- |
| `t_star`, `dias_reserva`, `horizonte`, `fraccion_seleccion` | Campos; `ValueError` si `dias_reserva` u `horizonte` no son enteros >= 1 o la fraccion no esta en (0, 1) |
| `de(corte, horizonte, fraccion_seleccion)` | Constructor desde el `ReserveCut` de L2 |
| `dias_seleccion` | `S = floor(f * R)` |
| `bloque(origen)` | `"seleccion"`, `"embargo"` o `"prueba"` |
| `origenes(bloque)` | Origenes diarios del bloque, en orden |
| `dias(bloque)` | Dias reservados que puntuan las ventanas de `"seleccion"` o `"prueba"` |
| `limites()` | Fraccion, dias, horizonte, `S` y `n`, primero y ultimo de `origenes_*` y `fechas_*`: lo que guarda el protocolo |

### `EvaluacionBloques`

`reserva`, `seleccion` y `prueba`, cada una `EvaluacionSku`, y las propiedades
`sku` y `sku_class`. Las vistas comparten las metricas por ventana; cambian los
agregados y los conteos.

### `Metricas`

| Campo | Tipo |
| --- | --- |
| `n_pares` | `int` |
| `mae`, `rmse`, `me` | `float` |
| `mase` | `float \| None` |
| `razon_sn` | `float \| None` (r frente a Seasonal Naive) |
| `no_calculables` | `Mapping[str, str]` metrica -> causa |

### `MetricasCandidato`

`candidato_id`, `familia`, `modelo`, `agregadas: Metricas | None`,
`por_ventana: tuple[MetricasVentana(origen, metricas), ...]`,
`n_ventanas_totales`, `n_ventanas_validas`, `n_pronosticos_validos`, y la
propiedad `cobertura = n_ventanas_validas / n_ventanas_totales`.

### `EvaluacionSku`

`sku`, `sku_class`, `linea_base: MetricasCandidato`,
`candidatos: tuple[MetricasCandidato, ...]`, `n_obs_validas_reserva`,
`reserva_toda_cero`, `n_candidatos_fallidos`.

## Seleccion y veredictos (`veredictos/`)

### `PoliticaSeleccion` (congelada, se declara antes de abrir la reserva)

| Campo | Default |
| --- | --- |
| `version` | `"3.4.4-bloques"` |
| `n_min` | `30` |
| `tolerancia_empate` | `0.01` |
| `orden_simplicidad` | `("classical", "ml", "dl", "foundation")` |
| `familias_elegibles` | `FAMILIAS_ELEGIBLES_INICIALES` |
| `alfa_dm` | `0.05` (nivel de HLN-DM con BH y de la confirmacion) |
| `n_min_dm` | `30` origenes comunes antes de emitir un p-valor o comparar un SKU |
| `fraccion_seleccion` | `0.4` (dias de la reserva del bloque de seleccion) |
| `n_min_confirmacion` | `10` SKUs minimos por competidora |
| `bloque_bootstrap` | `14` dias por bloque circular |
| `replicas_bootstrap` | `2000` |
| `semilla_bootstrap` | `20261029` |

La propiedad `confirmacion` da los `ParametrosConfirmacion` de la politica.
`ValueError` si la version esta vacia, `n_min < 1`, `n_min_dm < 2`,
`tolerancia_empate < 0`, `orden_simplicidad` repite familias,
`familias_elegibles` no cubre las cuatro categorias o alguna queda vacia o
repetida, `alfa_dm` o `fraccion_seleccion` no estan en (0, 1), alguno de
`n_min_confirmacion`, `bloque_bootstrap` o `replicas_bootstrap` no es un
entero >= 1, o `semilla_bootstrap` no es un entero >= 0.

### Funciones

```python
representante(evaluacion: EvaluacionSku, familia: str) -> MetricasCandidato | None
seleccionar_categoria(sku_class, evaluaciones, politica) -> SeleccionCategoria
emitir_veredictos(
    evaluaciones: Sequence[EvaluacionBloques],
    bloques: BloquesReserva,
    *,
    datos_sinteticos: bool | None,
    politica: PoliticaSeleccion = POLITICA_INICIAL,
    seleccion_previa: Sequence[SeleccionCategoria] = (),
) -> ResultadoEvaluacion
evaluacion_previa(sku, sku_class, t_star, historia, candidatos) -> EvaluacionSku
seleccionar_previa(previas, clases, politica=POLITICA_INICIAL) -> tuple[SeleccionCategoria, ...]
```

- `representante`: la configuracion de la familia con menor RMSE en la vista
  recibida. Con la vista `seleccion` es la instancia congelada del SKU.
- `seleccionar_categoria`: la regla de la categoria sobre las vistas que
  recibe; `emitir_veredictos` solo le pasa la vista `seleccion`.
- `emitir_veredictos`: selecciona en `seleccion` y emite todo lo demas con
  `prueba`. `EvaluacionRetrospectivaError` si un SKU se repite o si
  `bloques.fraccion_seleccion` no es la de la politica. `seleccion_previa` es
  la seleccion de robustez de cada categoria; una categoria ausente queda con
  `None`.
- `evaluacion_previa`: convierte los `CandidatoPrevio(candidato_id, familia,
  modelo, walk_forward)` de un SKU en una vista con un pliegue L3 por ventana,
  con Seasonal Naive ajustado en cada pliegue. `historia` es la serie
  admisible hasta t\* con que L3 armo los pliegues.
- `seleccionar_previa`: `seleccionar_categoria` sobre esas vistas, una por
  categoria de `clases` y en orden; una categoria sin vistas queda
  `sin_skus_comparables`.

### `SeleccionCategoria`

`sku_class`, `familia_campeona` (una familia o `"seasonal_naive"`),
`medianas_r: Mapping[str, float]`, `adverso: bool`,
`motivo` (`menor_mediana`, `empate_por_simplicidad`, `compuerta_linea_base`,
`sin_skus_comparables`), `skus_comparables`, `excluidos: Mapping[sku, causa]`
y `familias_excluidas: Mapping[familia, "no_entregada" | "no_elegible"]`.
Un SKU sin `n_min` ventanas comunes con la linea base para alguna familia queda
en `excluidos` con `pocas_ventanas:<familia>`.

### `VeredictoSku`

`sku`, `sku_class`, `veredicto`, `familia_campeona`, `candidato_campeon`,
`n_ventanas`, `cobertura`, `razon_sn`, `pierde_frente_a_linea_base`,
`comparacion_incompleta`, `metricas_campeon`, `metricas_linea_base`,
`iqr_diferencia_mae` (IQR de MAE campeon - MAE SN por ventana),
`diebold_mariano: PruebaDM`, `justificacion`, `modelos_evaluados`,
`intervalo_r: IntervaloRazon | None` (nivel `1 - alfa_dm`, solo con p-valor) e
`intervalo_r_fcr: IntervaloRazon | None` (nivel FCR de la categoria, solo si
BH selecciona al SKU). Todo se calcula en el bloque de prueba.

`modelos_evaluados: tuple[ModeloEvaluado(candidato_id, familia, modelo,
razon_sn), ...]` lista **todos** los modelos evaluados en el SKU con su r,
la linea base primero (su r vale 1) y luego los candidatos en el orden
recibido; `razon_sn` es `None` si no es calculable. Es solo descriptivo: la
seleccion definitiva sigue siendo por categoria (ADR-03-003).

### `ResumenCategoria` y `ResultadoEvaluacion`

- `SignificanciaCategoria`: `n_probados`, `n_mejores` y `n_peores` (BH, segun
  el signo del estadistico), `n_mejores_by`, `n_peores_by` (Benjamini-Yekutieli)
  y `nivel_fcr = 1 - R q / m`, `None` sin selecciones.
- `ResumenCategoria`: `sku_class`, `seleccion`, `conteos` y `porcentajes` por
  veredicto, `mediana_r`, `n_adversos` (SKUs con `r >= 1`), `significancia`,
  `confirmacion: ConfirmacionCategoria`, `contendiente` (la familia que la regla
  elige sin la fundacional, `None` si la fundacional no compite) y
  `seleccion_previa: SeleccionCategoria | None`.
- `ResultadoEvaluacion`: `version_politica`, `version_metricas`,
  `datos_sinteticos` (el origen con que se emitieron los veredictos),
  `categorias: tuple[ResumenCategoria, ...]`, `skus: tuple[VeredictoSku, ...]`
  y `bloques: BloquesReserva`.

## Diebold-Mariano (`diebold_mariano/prueba.py`)

```python
prueba_hln(diferencias: np.ndarray, horizonte: int) -> PruebaDM
probar_contra_linea_base(campeon, linea_base, *, n_min_ventanas=2) -> PruebaDM
probar_pares(a, b, *, n_min_ventanas=2) -> PruebaDM
intervalo_razon(campeon, linea_base, *, nivel: float) -> IntervaloRazon | None
corregir_multiplicidad(pruebas, alfa, *, metodo="fdr_bh") -> dict[str, PruebaDM]
```

- `prueba_hln`: DM con correccion HLN, t con `T - 1` grados de libertad y
  p-valor bilateral. `ValueError` si `horizonte < 1`.
- `probar_contra_linea_base`: diferencial `MSE_campeon - MSE_SN` por ventana
  comun (`rmse**2` de `por_ventana`); `horizonte` = mayor `n_pares` de esas
  ventanas.
- `probar_pares`: la misma prueba de `a` contra `b`; < 0 favorece a `a`. La
  confirmacion la usa entre familias.
- `emitir_veredictos` pasa el minimo de politica `n_min_dm` a la prueba.
  El default matematico de bajo nivel no sustituye ese minimo de corrida.
- `intervalo_razon`: los `rho >= 0` para los que HLN-DM no rechaza
  `L_campeon - rho^2 L_SN` al `nivel` dado (Fieller). Devuelve
  `IntervaloRazon(nivel, inferior, superior, inferior_cola)`. El conjunto es
  `[inferior, superior]`, o `[inferior, infinito)` con `superior = None`.
  Con `inferior_cola` tiene dos tramos, `[inferior, superior]` y
  `[inferior_cola, infinito)`, y la prueba rechaza los valores entre ambos.
  Los dos ultimos casos ocurren cuando la perdida de SN no se distingue de
  cero a ese nivel. `contiene(rho)` dice si la prueba no rechaza `rho`. Devuelve
  `None` si la prueba no se puede calcular. `ValueError` si `nivel` no esta en
  (0, 1).
- `corregir_multiplicidad`: Benjamini-Hochberg (`fdr_bh`) o
  Benjamini-Yekutieli (`fdr_by`) de `statsmodels` sobre las pruebas con
  p-valor; las demas quedan igual. `emitir_veredictos` la aplica dentro de cada
  categoria.

### `PruebaDM` (congelada)

| Campo | Tipo |
| --- | --- |
| `n_ventanas` | `int` (T) |
| `horizonte` | `int` (h) |
| `estadistico` | `float \| None`; < 0 = menor perdida del campeon |
| `p_valor` | `float \| None` (bilateral) |
| `p_ajustado` | `float \| None` (BH) |
| `significativa` | `bool \| None` (`p_ajustado <= alfa`) |
| `alfa` | `float \| None` (nivel usado en BH) |
| `causa` | `str \| None`: por que no se calculo |

Causas: `datos_sinteticos`, `campeon_es_linea_base`, `veredicto:<X>`,
`pocas_ventanas`, `varianza_no_positiva`, `sin_ventanas_comunes`.

## Confirmacion (`confirmacion/`)

```python
confirmar_campeones(
    categorias: Sequence[CategoriaAConfirmar],
    origenes: Iterable[pd.Timestamp],
    parametros: ParametrosConfirmacion,
    *,
    datos_sinteticos: bool | None,
) -> tuple[ConfirmacionCategoria, ...]
```

`origenes` son los origenes del bloque de prueba, en orden: el bootstrap
remuestrea sus posiciones. Holm ajusta entre las categorias con `p_iut`.

| Tipo | Campos |
| --- | --- |
| `ParametrosConfirmacion` | `alfa`, `n_min_ventanas`, `n_min_skus`, `bloque_dias`, `replicas`, `semilla` |
| `CategoriaAConfirmar` | `sku_class`, `familia_campeona`, `competidoras`, `instancias: Mapping[sku, Mapping[familia, MetricasCandidato]]` (prueba, linea base incluida) |
| `ConfirmacionCategoria` | `sku_class`, `familia_campeona`, `estado`, `causa`, `p_iut`, `p_holm`, `alfa`, `comparaciones`, `efecto_linea_base` |
| `ComparacionFamilia` | `competidora`, `n_skus`, `mediana_razon`, `ic_mediana`, `hodges_lehmann`, `ic_hodges_lehmann`, `n_victorias`, `ic_victorias`, `estadistico_wilcoxon`, `p_wilcoxon`, `p_iut`, `n_mejores`, `n_peores`, `causa` |
| `EfectoLineaBase` | `n_skus`, `mediana_r`, `ic_mediana`, `causa` |
| `Intervalo` | `inferior`, `superior` |

Las razones y sus cotas pueden valer 0 (solo el campeon no tiene error) o
infinito (solo la competidora, o Seasonal Naive, no lo tiene); dos pronosticos
sin error empatan con razon 1. Causa de una comparacion sin prueba:
`pocos_skus:n<minimo`, por ejemplo `pocos_skus:9<10`. Un valor que las perdidas
cero dejan sin definir queda en `None`, y `causa` de la comparacion o del efecto
lo nombra con `perdida_cero:<campos>`: `mediana_razon`, `ic_mediana` o
`hodges_lehmann` en `ComparacionFamilia`, y `mediana_r` o `ic_mediana` en
`EfectoLineaBase`. En una comparacion sin prueba las dos causas se unen con
`; `. Causas de la categoria: `pocos_skus:<familias>`,
`sin_familias_competidoras`, `campeon_es_linea_base`, `datos_sinteticos`,
`origen_desconocido` y, para `no_confirmado`, `p_holm><alfa>` y
`intervalo_no_bajo_1:<familias>`, unidas con `; `.

## Excepciones

```
EvaluacionRetrospectivaError(ValueError)
```

Se lanza si un pronostico tiene origen antes de t\* o fechas dentro de la
historia, si un origen se repite en un candidato, si una ventana no son los
dias que siguen a su origen o cambia entre candidatos, si un SKU llega dos
veces a `emitir_veredictos`, si los bloques son de otra reserva o si su
fraccion no es la de la politica.

## Logging

- `ERROR` antes de cada `EvaluacionRetrospectivaError` (`sku`, `candidato_id`).
- `WARNING` si la linea base no se puede ajustar (historia corta).
- `INFO` por SKU evaluado (ventanas validas de la linea base), por categoria
  seleccionada (campeona, motivo, comparables, excluidos, familias excluidas),
  por cada correccion de Diebold-Mariano (pruebas y significativas) y por
  categoria confirmada (campeona, estado, p de Holm y causa).
- Sin series de demanda ni PII en los mensajes.
