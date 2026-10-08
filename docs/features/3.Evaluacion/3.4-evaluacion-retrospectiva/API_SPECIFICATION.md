# API — Evaluacion 3.4 (Evaluacion retrospectiva)

Identificadores en espanol; comentarios y mensajes en espanol. 3.4 recibe
arreglos y fechas: no ajusta modelos, no lee ni escribe archivos.

Puntos de entrada:

- Metricas (A1): `pred_engine.forecasting.evaluaciones.calculo_errores`
- Seleccion y veredictos (A2): `pred_engine.forecasting.evaluaciones.veredictos`

## Constantes

| Nombre | Valor |
| --- | --- |
| `VERSION_METRICAS` | `"3.4.0"` |
| `PERIODO_ESCALA` | `7` (m de Q1, igual que Seasonal Naive) |
| `FAMILIA_LINEA_BASE` | `"seasonal_naive"` |
| `POLITICA_INICIAL` | `PoliticaSeleccion()` (ver abajo) |
| `VEREDICTOS` | `FALLO_TECNICO`, `NO_EVALUABLE`, `EVIDENCIA_INSUFICIENTE`, `VALIDADO`, `EXPLORATORIO` |

## Entrada (`calculo_errores/contratos.py`)

Dataclasses congeladas (`eq=False` las que llevan arreglos).

### `PronosticoFechado`

| Campo | Tipo | Regla |
| --- | --- | --- |
| `origen` | `pd.Timestamp` | `>= t*` |
| `fechas` | `pd.DatetimeIndex` | todas `> t*` |
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
| `linea_base` | `SerieCandidato` (Seasonal Naive) |
| `candidatos` | `tuple[SerieCandidato, ...]` |
| `n_candidatos_fallidos` | `int` (default 0): candidatos que no llegaron a pronosticar |

Para la salida actual del pipeline: un `PronosticoFechado(origen=t*,
fechas=reserve.reserved_dates, valores=FittedCandidate.forecast)` por candidato.

## Metricas (`calculo_errores/metricas.py`)

```python
separar_reserva(serie: pd.Series, corte: ReserveCut) -> tuple[np.ndarray, pd.Series]
escala_q1(historia: np.ndarray, m: int = 7) -> tuple[float | None, str | None]
evaluar_sku(entrada: EntradaSku, corte: ReserveCut) -> EvaluacionSku
```

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
| `version` | `"3.4.0-inicial"` |
| `n_min` | `30` |
| `tolerancia_empate` | `0.01` |
| `orden_simplicidad` | `("classical", "ml", "dl", "foundation")` |

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
`sin_skus_comparables`), `skus_comparables`, `excluidos: Mapping[sku, causa]`.

### `VeredictoSku`

`sku`, `sku_class`, `veredicto`, `familia_campeona`, `candidato_campeon`,
`n_ventanas`, `cobertura`, `razon_sn`, `pierde_frente_a_linea_base`,
`comparacion_incompleta`, `metricas_campeon`, `metricas_linea_base`,
`iqr_diferencia_mae` (IQR de MAE campeon - MAE SN por ventana),
`justificacion` y `modelos_evaluados`.

`modelos_evaluados: tuple[ModeloEvaluado(candidato_id, familia, modelo,
razon_sn), ...]` lista **todos** los modelos evaluados en el SKU con su r,
la linea base primero (su r vale 1) y luego los candidatos en el orden
recibido; `razon_sn` es `None` si no es calculable. Es solo descriptivo: la
seleccion definitiva sigue siendo por categoria (ADR-03-003).

### `ResumenCategoria` y `ResultadoEvaluacion`

- `ResumenCategoria`: `sku_class`, `seleccion`, `conteos` y `porcentajes` por
  veredicto, `mediana_r` y `n_adversos` (SKUs con `r >= 1`).
- `ResultadoEvaluacion`: `version_politica`, `version_metricas`,
  `categorias: tuple[ResumenCategoria, ...]` y `skus: tuple[VeredictoSku, ...]`.
  El origen de los datos no se guarda aqui: es un dato de la corrida (3.5).

## Excepciones

```
EvaluacionRetrospectivaError(ValueError)
```

Se lanza si un pronostico tiene origen antes de t\* o fechas dentro de la
historia, si un origen se repite en un candidato, o si un SKU llega dos veces
a `emitir_veredictos`.

## Logging

- `ERROR` antes de cada `EvaluacionRetrospectivaError` (`sku`, `candidato_id`).
- `INFO` por SKU evaluado (ventanas validas de la linea base) y por categoria
  seleccionada (campeona, motivo, comparables, excluidos).
- Sin series de demanda ni PII en los mensajes.
