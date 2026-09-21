# API — Seleccion 2.4 (HPO: TPE Sampler + ASHA Resource Allocator)

Identificadores en espanol; comentarios y mensajes en espanol. Solo
`adaptador_optuna.py` importa Optuna (ver docstring del archivo).

## Espacio de busqueda (`espacio.py`)

### Tipos de parametro

| Tipo | Campos | Codificacion |
| --- | --- | --- |
| `Entero(nombre, bajo, alto, paso=1)` | rango entero | directo |
| `Flotante(nombre, bajo, alto, log=False)` | rango flotante, escala log opcional | directo |
| `Categorico(nombre, opciones)` | universo sin orden | directo (`suggest_categorical`) |
| `Ordinal(nombre, niveles)` | universo **con orden** | indice entero `0..len(niveles)-1` (`suggest_int`); decodificar con `nivel_de(parametro, indice)` |

`Parametro = Entero | Flotante | Categorico | Ordinal`.

### `Condicion(parametro, padre, valores)`

Dependencia jerarquica: `parametro` solo se muestrea si `configuracion[padre]
in valores`. El padre debe declararse antes en `EspacioBusqueda.parametros`.

### `EspacioBusqueda(parametros, condiciones=(), restricciones=())`

| Metodo | Rol |
| --- | --- |
| `muestrear(rng, intentos_max=200)` | Muestra local (fuera de Optuna), respeta condiciones/restricciones |
| `es_valida(configuracion)` | Evalua `restricciones` |
| `nombres_activos(configuracion)` | Parametros activos segun condiciones |
| `descripcion_canonica()` | Dict JSON-serializable (participa en la huella de 2.9) |

`descripcion_canonica()["parametros"]` incluye, por tipo:
`{"tipo": "ordinal", "nombre": ..., "niveles": [...]}` para `Ordinal`
(analogo a `"entero"`/`"flotante"`/`"categorico"`).

## Muestreadores (`muestreadores.py`)

| Funcion | Rol |
| --- | --- |
| `construir_muestreador_tpe(seed=0, n_arranque=10, n_candidatos=24, multivariate=True)` | `optuna.samplers.TPESampler` (ADR-012) |
| `construir_muestreador_aleatorio(seed=0)` | `optuna.samplers.RandomSampler` |

## ASHA (`asha.py`, `poda.py`)

### `ReglasPoda` (dataclass, `frozen=True, slots=True`)

| Campo | Default | Rol |
| --- | --- | --- |
| `min_ventanas` | `4` | Regla de seguridad #1: sin poda antes de esto |
| `agregacion` | `"media_recortada"` | `"media"` \| `"mediana"` \| `"media_recortada"` |
| `proporcion_recorte` | `0.1` | Solo si `agregacion="media_recortada"` |
| `factor_reduccion` | `3` | Retiene el mejor `1/factor_reduccion` por escalon |
| `habilitar_poda_semantica` | `True` | Activa `es_degenerada` |

### `es_degenerada(y_pred, y_train, tol=1e-9) -> (bool, str | None)`

Regla de seguridad #3. Detecta: prediccion no finita, demanda negativa,
prediccion nula cuando el entrenamiento no lo era.

### `DecisorASHA(reglas, n_ventanas_totales)`

Algoritmo puro, sin Optuna.

| Metodo | Rol |
| --- | --- |
| `escalones()` | Tupla de escalones geometricos (`min_ventanas`, `*factor_reduccion`, ..., `n_ventanas_totales`) |
| `decidir(n_evaluadas, numero_trial, valores_competidores_en_escalon)` | `DecisionPoda(podar, motivo)` |

Compara solo contra competidores del **mismo escalon** (ver ADR-013 para la
diferencia con la redaccion literal de Notion sobre "percentil 75 de todas
las configuraciones").

## Contratos (`contratos.py`)

`Protocol`s de frontera; ningun `import optuna` cruza hacia aqui.

| Nombre | Rol |
| --- | --- |
| `TrialHPO` | `number`, `suggest_int/float/categorical`, `set_user_attr`, `report`, `should_prune` |
| `InfoTrial` | Snapshot de un trial cerrado: `numero`, `estado`, `valor`, `parametros`, `atributos` |
| `EstudioHPO` | `ask()`, `tell(...)`, `trials_finalizados()`, `agregar_trials_historicos(historico, espacio)` |
| `ProveedorMotivoPoda` | `motivo_de(trial_number) -> str \| None` |

`EstudioHPO.agregar_trials_historicos` es nuevo en esta sesion (warm start,
ver mas abajo); implementado por `_EstudioOptuna` en `adaptador_optuna.py`.

## Warm start (`adaptador_optuna.py`)

Distinto de reanudar una corrida interrumpida (misma corrida, mismo
`run_id`, ver 2.9): importa evidencia de **otra** corrida.

| Funcion | Rol |
| --- | --- |
| `semillas_desde_historico(historico, *, espacio) -> list[FrozenTrial]` | Traduce `InfoTrial` externo a `FrozenTrial`; descarta (no aborta) trials con parametros/valores incompatibles con `espacio` |
| `inyectar_historico(study, semillas) -> int` | `study.add_trial()` por semilla; retorna cuantas se inyectaron |

Uso end-to-end: `ejecutar_estudio(..., historico_previo=historico)` inyecta
antes del primer trial (solo si `indice_trial == 0`, es decir, corrida
nueva o sin controlador -- no se reinyecta al reanudar).

## `ejecutar_estudio` (`estudio.py`) — cambios de esta sesion

| Parametro nuevo | Tipo | Default | Efecto |
| --- | --- | --- | --- |
| `historico_previo` | `Sequence[InfoTrial] \| None` | `None` | Warm start (ver arriba) |

Cada trial ahora registra `trial.set_user_attr("timestamp", ...)` (ISO-8601
UTC) ademas de `trial_id`/`familia`/`sku_id`. Se propaga a
`Trial.timestamp` (`comun/dataclasses/hpo.py`) via `registro.py`.

## `Trial` (`comun/dataclasses/hpo.py`) — campo nuevo

| Campo nuevo | Tipo | Default |
| --- | --- | --- |
| `timestamp` | `str \| None` | `None` |

## Reanudacion con `TPESampler` — limitacion conocida (ADR-012)

`adaptador_optuna._sincronizar_sampler_aleatorio` solo resincroniza el RNG
de `RandomSampler` (ver 2.9). Con `TPESampler`, ahora emite
`_logger.warning(...)` en vez de fallar silenciosamente: las primeras
propuestas de una corrida reanudada pueden repetir configuraciones ya
evaluadas antes de la interrupcion. Los trials previos a la interrupcion
se preservan intactos; lo que no se garantiza es que los trials *nuevos*
coincidan con los de una corrida continua (ver
`test_reanudacion_con_tpe_preserva_los_trials_ya_evaluados`).

## `classical_selection.py` — cableo de reanudacion

| Parametro nuevo | Tipo | Default |
| --- | --- | --- |
| `raiz_corrida` (en `seleccionar_configuracion_clasica`) | `str \| Path \| None` | `None` |
| `run_id` (en `seleccionar_configuracion_clasica`) | `str \| None` | `None` |

`seleccionar_por_panel(**kwargs)` **rechaza** `run_id` explicito
(`ValueError`): un `run_id` literal colisionaria entre SKUs. Para
persistencia en modo panel, llamar `seleccionar_configuracion_clasica` por
SKU con un `run_id` propio (convencion sugerida:
`{familia}-{sku_id}-{sesion}`, ver `GUIA_CICLO_VIDA_CORRIDA.md`).

## Benchmarks (`benchmarks/`)

| Modulo | Rol |
| --- | --- |
| `funciones_objetivo.py` | `sphere`, `rastrigin`, `rosenbrock` — `Mapping[str, float] -> float`, minimo global 0.0 |
| `comparador_convergencia.py` | `comparar_convergencia(funcion, ..., dim, bajo, alto, n_trials, seed) -> ResultadoConvergencia` |

`ResultadoConvergencia`: `nombre_funcion`, `mejor_tpe`, `mejor_aleatorio`,
propiedad `tpe_converge_mejor`.

## Logging

`pred_engine.comun.logger.get_logger(__name__)` en `adaptador_optuna.py` y
`estudio.py`. Eventos nuevos: warm start inyectado (`INFO`), reanudacion con
`TPESampler` (`WARNING`, no bloqueante).
