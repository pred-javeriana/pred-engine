# API — Evaluacion 3.4 (Evaluacion retrospectiva)

Identificadores en espanol; comentarios y mensajes en espanol. De los
candidatos 3.4 recibe arreglos y fechas: no los ajusta, no lee ni escribe
archivos. El unico modelo que ajusta es la linea base Seasonal Naive.

Puntos de entrada:

- Metricas y linea base (A1): `pred_engine.forecasting.evaluaciones.calculo_errores`
- Seleccion y veredictos (A2): `pred_engine.forecasting.evaluaciones.veredictos`
- Diebold-Mariano: `pred_engine.forecasting.evaluaciones.diebold_mariano`

## Constantes

| Nombre | Valor |
| --- | --- |
| `VERSION_METRICAS` | `"3.4.1"` |
| `PERIODO_ESCALA` | `7` (m de Q1, igual que Seasonal Naive) |
| `FAMILIA_LINEA_BASE` | `"seasonal_naive"` |
| `MODELO_LINEA_BASE` | `"seasonal_naive"` |
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
pronosticar_linea_base(entrada: EntradaSku, corte: ReserveCut) -> SerieCandidato
evaluar_sku(entrada: EntradaSku, corte: ReserveCut) -> EvaluacionSku
```

- `pronosticar_linea_base`: Seasonal Naive (`instanciar_linea_base()` de 3.2)
  ajustado una vez con `historia` y pronosticado en cada ventana de los
  candidatos con `pronosticar(observadas, H)`. Ventana sin valores finitos si
  no se puede ajustar o falta una observacion antes del origen.
- `evaluar_sku`: genera la linea base y calcula las metricas de ella y de cada
  candidato.

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
| `version` | `"3.4.1-inicial"` |
| `n_min` | `30` |
| `tolerancia_empate` | `0.01` |
| `orden_simplicidad` | `("classical", "ml", "dl", "foundation")` |
| `familias_elegibles` | `FAMILIAS_ELEGIBLES_INICIALES` |
| `alfa_dm` | `0.05` (nivel de HLN-DM con BH) |

`ValueError` si la version esta vacia, `n_min < 1`, `tolerancia_empate < 0`,
`orden_simplicidad` repite familias, `familias_elegibles` no cubre las cuatro
categorias o alguna queda vacia o repetida, o `alfa_dm` no esta en (0, 1).

### Funciones

```python
representante(evaluacion: EvaluacionSku, familia: str) -> MetricasCandidato | None
seleccionar_categoria(sku_class, evaluaciones, politica) -> SeleccionCategoria
emitir_veredictos(
    evaluaciones: Sequence[EvaluacionSku],
    *,
    datos_sinteticos: bool,
    politica: PoliticaSeleccion = POLITICA_INICIAL,
) -> ResultadoEvaluacion
```

### `SeleccionCategoria`

`sku_class`, `familia_campeona` (una familia o `"seasonal_naive"`),
`medianas_r: Mapping[str, float]`, `adverso: bool`,
`motivo` (`menor_mediana`, `empate_por_simplicidad`, `compuerta_linea_base`,
`sin_skus_comparables`), `skus_comparables`, `excluidos: Mapping[sku, causa]`
y `familias_excluidas: Mapping[familia, "no_entregada" | "no_elegible"]`.

### `VeredictoSku`

`sku`, `sku_class`, `veredicto`, `familia_campeona`, `candidato_campeon`,
`n_ventanas`, `cobertura`, `razon_sn`, `pierde_frente_a_linea_base`,
`comparacion_incompleta`, `metricas_campeon`, `metricas_linea_base`,
`iqr_diferencia_mae` (IQR de MAE campeon - MAE SN por ventana),
`diebold_mariano: PruebaDM`, `justificacion` y `modelos_evaluados`.

`modelos_evaluados: tuple[ModeloEvaluado(candidato_id, familia, modelo,
razon_sn), ...]` lista **todos** los modelos evaluados en el SKU con su r,
la linea base primero (su r vale 1) y luego los candidatos en el orden
recibido; `razon_sn` es `None` si no es calculable. Es solo descriptivo: la
seleccion definitiva sigue siendo por categoria (ADR-03-003).

### `ResumenCategoria` y `ResultadoEvaluacion`

- `ResumenCategoria`: `sku_class`, `seleccion`, `conteos` y `porcentajes` por
  veredicto, `mediana_r` y `n_adversos` (SKUs con `r >= 1`).
- `ResultadoEvaluacion`: `version_politica`, `version_metricas`,
  `datos_sinteticos` (el origen con que se emitieron los veredictos),
  `categorias: tuple[ResumenCategoria, ...]` y `skus: tuple[VeredictoSku, ...]`.

## Diebold-Mariano (`diebold_mariano/prueba.py`)

```python
prueba_hln(diferencias: np.ndarray, horizonte: int) -> PruebaDM
probar_contra_linea_base(campeon: MetricasCandidato, linea_base: MetricasCandidato) -> PruebaDM
corregir_multiplicidad(pruebas: Mapping[str, PruebaDM], alfa: float) -> dict[str, PruebaDM]
```

- `prueba_hln`: DM con correccion HLN, t con `T - 1` grados de libertad y
  p-valor bilateral. `ValueError` si `horizonte < 1`.
- `probar_contra_linea_base`: diferencial `MSE_campeon - MSE_SN` por ventana
  comun (`rmse**2` de `por_ventana`); `horizonte` = mayor `n_pares` de esas
  ventanas.
- `corregir_multiplicidad`: Benjamini-Hochberg (`statsmodels`, `fdr_bh`) sobre
  las pruebas con p-valor; las demas quedan igual. `emitir_veredictos` la
  aplica sobre todos los SKUs de la corrida.

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

## Excepciones

```
EvaluacionRetrospectivaError(ValueError)
```

Se lanza si un pronostico tiene origen antes de t\* o fechas dentro de la
historia, si un origen se repite en un candidato, si una ventana no son los
dias que siguen a su origen o cambia entre candidatos, o si un SKU llega dos
veces a `emitir_veredictos`.

## Logging

- `ERROR` antes de cada `EvaluacionRetrospectivaError` (`sku`, `candidato_id`).
- `WARNING` si la linea base no se puede ajustar (historia corta).
- `INFO` por SKU evaluado (ventanas validas de la linea base), por categoria
  seleccionada (campeona, motivo, comparables, excluidos, familias excluidas)
  y por corrida de Diebold-Mariano (pruebas y significativas).
- Sin series de demanda ni PII en los mensajes.
