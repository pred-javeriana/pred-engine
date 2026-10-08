# API — Evaluacion 3.2 (Adaptador del conjunto de candidatos de M2)

Identificadores, comentarios y mensajes en espanol. 3.2 es una transformacion
pura: recibe el manifiesto JSON y devuelve candidatos validados, fallos y
adaptadores sin ajustar. No ajusta, no evalua y no persiste.

Puntos de entrada:

- Esquema compartido: `pred_engine.comun.modelos.manifiesto_candidatos`
- Productor de M2: `pred_engine.optimizacion.router.construir_manifiesto`
- Adaptador de M3: `pred_engine.forecasting.adaptador_candidatos`

## Esquema compartido (`comun/modelos/manifiesto_candidatos.py`)

Modelos Pydantic estrictos (`strict`, `extra="forbid"`, congelados, sin
`inf`/`nan`).

| Nombre | Contenido |
| --- | --- |
| `VERSION_ESQUEMA_MANIFIESTO` | `1` |
| `MODELO_POR_FAMILIA` | `classical -> sarima`, `ml -> lightgbm`, `dl -> mlp`, `foundation -> chronos2` |
| `ContextoParticion` | `ingesta_ref_m1: str`, `t_corte_reserva: date`, `fraccion_reserva: float` en (0, 1) |
| `ManifiestoCandidatos` | `schema_version`, `run_id_m2`, `emitido_en: AwareDatetime`, `contexto`, `candidatos: tuple[dict, ...]` (crudos) |
| `Candidato` | union discriminada por `familia`: `CandidatoClasico`, `CandidatoML`, `CandidatoDL`, `CandidatoFundacional` |
| `configuracion_declarada(familia, forecast_config)` | la configuracion que M2 declara; para `foundation`, la de `CHRONOS2_ZERO_SHOT` |

Cada candidato lleva `candidato_id`, `sku`, `sku_class`, `familia`, `modelo`,
`semilla` (>= 0) y su `configuracion`:

| Familia | Configuracion (todos los campos obligatorios) |
| --- | --- |
| `classical` | `ConfigSarima`: `p, d, q, P, D, Q, m, tendencia`. `m` es 0 (sin estacionalidad, entonces `P = D = Q = 0`) o >= 2; `tendencia` es `"c"` solo si `d = D = 0` (ADR-018) |
| `ml` | `ConfigLightGBM`: `lags, m, n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda` |
| `dl` | `ConfigMLP`: `lags, capas (>= 2), unidades, epochs, batch_size, learning_rate, dropout, l2` |
| `foundation` | `ConfigChronos2`: cada campo debe ser igual al de `CHRONOS2_ZERO_SHOT` (revision de pesos, cuantil 0.5) |

## Productor de M2 (`optimizacion/router/manifiesto.py`)

```python
construir_manifiesto(
    resultados: Sequence[SelectionResult],
    *,
    run_id_m2: str,
    contexto: ContextoParticion,
    emitido_en: datetime,
) -> str  # JSON
```

- Un candidato por `SelectionResult`, con `forecast_config` y
  `forecast_seed` (no `payload`).
- `candidato_id = "{sku}/{familia}/{modelo}"`.
- `SelectionContractError` si un resultado no trae `forecast_config`.

## Adaptador de M3 (`forecasting/adaptador_candidatos/`)

### Funciones

```python
validar_manifiesto(
    contenido: str | bytes,
    *,
    esperado: ContextoParticion,
    skus_panel: Collection[str],
    reserva_abierta_en: datetime | None = None,
    fabricas: Mapping[str, FabricaAdaptador] = FABRICAS,
    cargar_fundacional: Callable[[ConfiguracionFundacional], object] = cargar_pipeline,
) -> HandoffValidado
instanciar(
    candidato: Candidato,
    *,
    fabricas: Mapping[str, FabricaAdaptador] = FABRICAS,
    pipeline_fundacional: PipelineFundacional | None = None,
) -> AdaptadorCandidato
instanciar_linea_base() -> AdaptadorCandidato  # SeasonalNaiveStub(season_length=7)
```

- `validar_manifiesto` valida primero el lote y luego cada candidato. Con
  candidatos fundacionales validos carga una vez los pesos con
  `cargar_fundacional`. `fabricas` permite simular una familia deshabilitada.
- `instanciar` pasa a la fabrica la configuracion validada completa y la
  semilla; para `foundation` usa la configuracion congelada y el pipeline
  inyectado. `ModeloNoRegistradoError` si la familia no tiene fabrica.

### Constantes y contratos

| Nombre | Contenido |
| --- | --- |
| `FABRICAS` | `classical -> fabrica_sarima`, `ml -> fabrica_ml`, `dl -> fabrica_dl`, `foundation -> fabrica_fundacional` (solo lectura) |
| `PERIODO_LINEA_BASE` | `7` |
| `AdaptadorCandidato` | protocolo: `fit(y) -> AdaptadorCandidato` y `pronosticar(observadas, horizon) -> np.ndarray` |
| `FabricaAdaptador` | `(configuracion: Mapping, *, seed: int) -> AdaptadorCandidato` |
| `HandoffValidado` | `contexto`, `run_id_m2`, `candidatos: tuple[Candidato, ...]`, `fallos: tuple[FalloCandidato, ...]` |
| `FalloCandidato` | `sku`, `candidato_id`, `modelo: str \| None`, `campo`, `codigo_error: CodigoFallo`, `mensaje` (nombra SKU, candidato, modelo y campo) |
| `MotivoRechazo` | `manifiesto_ilegible`, `version_no_soportada`, `contexto_incompatible`, `handoff_posterior_a_reserva`, `candidato_id_duplicado`, `sku_fuera_del_panel`, `error_no_clasificado` |
| `CodigoFallo` | `modelo_no_registrado`, `campo_faltante`, `campo_sobrante`, `tipo_incorrecto`, `fuera_de_dominio`, `semilla_ausente`, `pesos_no_disponibles` |

### `pronosticar` (`forecasting/base.py`)

```python
BaseForecaster.pronosticar(observadas: np.ndarray, horizon: int) -> np.ndarray
```

- `observadas`: valores reales que siguen a la serie de `fit` hasta el origen;
  1D y finitos, si no `ValueError`. Vacio, el origen es el fin de esa serie y
  el resultado es el de `predict(horizon)`.
- No cambia lo ajustado: cada origen se pronostica aparte.
- Lo implementan SARIMA, LightGBM, MLP, Chronos-2 y Seasonal Naive.

## Excepciones (`errores.py`)

```
AdaptadorCandidatosError(Exception)
├── LoteRechazadoError(motivo: MotivoRechazo, detalle: str)
└── ModeloNoRegistradoError
```

## Logging

- `ERROR` por cada lote rechazado (`motivo`, `detalle`) y por cada candidato
  rechazado (`sku`, `candidato_id`, `modelo`, `campo`, `codigo`).
- `INFO` al validar el handoff (`run_id_m2`, candidatos, fallos) y al emitir
  el manifiesto en M2.
- Sin series de demanda ni PII en los mensajes.
