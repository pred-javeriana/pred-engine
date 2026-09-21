# API — Seleccion 2.5 (MLSelectionStrategy)

Identificadores en espanol; comentarios y mensajes en espanol. La familia ML
no importa Optuna: toda la optimizacion pasa por `ejecutar_estudio` (2.4).

## Modelo (`comun/modelos/modelos_machine_learning/lgbm.py`)

### `LightGBMForecaster(BaseForecaster)`

Satisface `Pronosticador` (`fit(y)` / `predict(horizon)`).

| Argumento (solo keyword) | Default | Rol |
| --- | --- | --- |
| `lags` | `7` | Observaciones previas por muestra (>= 1) |
| `estacionalidad` | `1` | `m`; si `> 1` agrega la fase `posicion mod m` |
| `n_estimators` | `100` | Rondas de boosting |
| `max_depth` | `4` | Profundidad maxima (`num_leaves = min(31, 2**max_depth)`) |
| `learning_rate` | `0.1` | |
| `min_child_weight` | `1.0` | `min_sum_hessian_in_leaf` |
| `subsample` | `1.0` | `bagging_fraction` (activa `bagging_freq=1` si `< 1`) |
| `colsample_bytree` | `1.0` | `feature_fraction` |
| `reg_alpha`, `reg_lambda` | `0.0` | `lambda_l1`, `lambda_l2` |
| `seed` | `0` | Determinismo |
| `forzar_no_negativo` | `True` | Recorta el pronostico a >= 0 |

- `fit(y)`: exige 1D finito y `len(y) >= lags + MUESTRAS_MINIMAS` (5); si no,
  `SerieCortaError` (`ValueError`). Fallos de LightGBM -> `AjusteModeloError`.
- `predict(horizon)`: recursivo; `RuntimeError` antes de `fit`, `ValueError`
  si `horizon < 1`.
- Determinismo: `num_threads=1`, `deterministic=True`, `force_row_wise=True`.

### `construir_muestras(serie, lags, estacionalidad) -> (X, objetivos)`

Fila `i` usa `serie[i : i + lags]` y predice `serie[i + lags]`; X tiene
`lags + 3 [+ 1]` columnas (lags, media, desviacion, fraccion de ceros, fase).

### `fabrica_ml(configuracion, *, seed) -> LightGBMForecaster`

`FabricaPronosticador`. Claves ausentes usan el default; `m` (no buscado) la
inyecta el llamador y se mapea a `estacionalidad`.

## Espacio (`optimizadores/modelos_machine_learning/espacio_ml.py`)

`EspacioML` (`frozen=True, slots=True`), campos con sus defaults:
`lags_min=3, lags_max=14, n_estimators_min=20, n_estimators_max=300,
max_depth_min=2, max_depth_max=8, learning_rate_min=0.01,
learning_rate_max=0.3, min_child_weight_min=1.0, min_child_weight_max=20.0,
subsample_min=0.5, colsample_min=0.5, reg_min=1e-8, reg_max=10.0,
costo_max=1500, m=7`.

| Funcion | Rol |
| --- | --- |
| `construir_espacio_ml(espacio=None) -> EspacioBusqueda` | 9 parametros (`lags`, `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`); `Flotante(log=True)` en `learning_rate`, `min_child_weight`, `reg_*`; restriccion `n_estimators * max_depth <= costo_max` |
| `min_train_recomendado_ml(espacio=None) -> int` | `lags_max + max(2*m, 10)` (con `m>1`) |

## Seleccion (`optimizadores/modelos_machine_learning/ml_selection.py`)

### `seleccionar_configuracion_ml(y, sku_id=None, espacio=None, n_trials=30, min_train=None, horizonte=7, paso=7, metrica_objetivo="mase", muestreador=None, reglas=None, seed=0, raiz_corrida=None, run_id=None, historico_previo=None) -> ResultadoSeleccionML`

- `ValueError` si `y` no es 1D o `len(y) < min_train + horizonte`
  (mensaje con el `sku_id`).
- Llama a `seleccionar_con_hpo(..., familia="ml")`.
- `raiz_corrida` + `run_id` activan persistencia y reanudacion (2.9);
  `historico_previo` es warm start (2.4).

### `seleccionar_por_panel_ml(panel, *, n_procesos=None, **kwargs) -> dict[str, ResultadoSeleccionML]`

Un estudio por SKU, ordenado alfabeticamente. `run_id` en kwargs -> `ValueError`.
`n_procesos > 1` reparte SKUs con `spawn`; resultado identico al secuencial.

### Dataclasses (`comun/dataclasses/modelos_machine_learning.py`)

| Nombre | Campos |
| --- | --- |
| `ConfiguracionSeleccionadaML` | `sku_id`, `hiperparametros: Mapping`, `metrica_objetivo`, `valor`, `n_ventanas` |
| `ResultadoSeleccionML` | `seleccionada \| None`, `estudio: ResultadoEstudio`; propiedad `descartadas` (podados + fallidos) |

## Estrategia (`optimizadores/modelos_machine_learning/estrategia.py`)

### `MLSelectionStrategy` (`family = "ml"`)

`__init__(*, espacio=None, presupuestos=None, horizonte=7, paso=7, metrica_objetivo="mase", min_train=None, seed=0, raiz_corrida=None, sesion=None)`

- `raiz_corrida` sin `sesion` -> `ValueError`. Con ambos, `run_id =
  ml-{sku_id}-{sesion}` (estable entre interrupcion y reanudacion).
- `select(request, profile) -> SelectionResult`:
  1. Perfil sin presupuesto -> `SelectionContractError`.
  2. Sin serie -> `SelectionContractError`.
  3. Ordena la serie por `timestamp`, llama a `seleccionar_configuracion_ml`.
  4. `ValueError` de la seleccion (serie corta, etc.) -> `SelectionContractError`
     con la causa encadenada.
  5. Sin ningun trial completado -> `SelectionContractError`.
- `SelectionResult.payload`: `hiperparametros`, `metrica_objetivo`, `valor`,
  `n_ventanas`, `n_trials`, `n_completados`, `n_podados`, `n_fallidos`, `seed`.
  `produced_by = "MLSelectionStrategy"`; el router anota `policy_version`.

### `PresupuestoHPO(n_trials, reglas=ReglasPoda())` y `PRESUPUESTO_POR_PERFIL`

`dense_stable` 30, `dense_variable` 40, `sparse_stable` 25,
`sparse_variable` 25 trials.

## Nucleo compartido (`optimizadores/seleccion_hpo.py`)

| Funcion | Rol |
| --- | --- |
| `exigir_serie(y, *, min_train, horizonte, sku_id)` | 1D + longitud minima |
| `seleccionar_con_hpo(serie, espacio, fabrica, *, familia, sku_id, n_trials, min_train, horizonte, paso, metrica_objetivo, estacionalidad, muestreador, reglas, seed, raiz_corrida, run_id, historico_previo)` | Un estudio por SKU; delega en `ejecutar_estudio` |
| `series_por_sku(panel)` | Una serie 1D por SKU, orden alfabetico, ordenada por `timestamp` |
| `seleccionar_panel(panel, seleccionar_una, *, n_procesos=None, **kwargs)` | Reparto por SKU; `seleccionar_una` de nivel de modulo (pickle); pool con contexto `spawn` |

## Benchmark (`.../benchmarks/`)

| Funcion | Rol |
| --- | --- |
| `serie_sintetica(clase, n=200, seed=0)` | `smooth` / `erratic` / `intermittent` (otra clase -> `ValueError`) |
| `comparar_con_baseline(clase, *, seed=0, n=200, holdout=28, n_trials=25) -> ResultadoBenchmarkML` | MAE del holdout: seleccionada vs default vs ingenuo estacional; incluye `ahorro_asha` |
| `ahorro_asha(estudio) -> float` | 1 - ventanas evaluadas / ventanas sin poda |

## Logging

`pred_engine.comun.logger.get_logger(__name__)`. La estrategia registra un
`INFO` por seleccion (`sku_id`, perfil, metrica, trials, podados). Sin series
de demanda en los logs.
