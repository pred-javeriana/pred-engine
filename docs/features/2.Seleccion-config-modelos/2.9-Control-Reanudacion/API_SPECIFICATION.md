# API — Seleccion 2.9 (Control de reanudacion)

Identificadores en espanol; comentarios y mensajes en espanol. El paquete
`control_reanudacion` no importa Optuna, Walk-Forward ni modelos concretos.

Punto de entrada generico: `pred_engine.optimizacion.control_reanudacion`.
Integracion HPO: `pred_engine.optimizacion.optimizadores.HPO.estudio`.

## Layout en disco

```
{raiz_corrida}/
  {run_id}/
    manifiesto.json      # ManifiestoCorrida (atomico)
    backend.jsonl        # trials Optuna finalizados (atomico)
```

- `AlmacenManifiestosFs(raiz_corrida)` resuelve
  `{raiz_corrida}/{run_id}/manifiesto.json`.
- `AdaptadorPersistenciaOptuna` escribe
  `{raiz_corrida}/{run_id}/backend.jsonl`.
- `backend_checkpoint` en el manifiesto guarda el nombre del archivo
  (`backend.jsonl`), no una ruta absoluta.

## Constantes

| Nombre | Valor |
| --- | --- |
| `SCHEMA_VERSION` | `1` |
| `BackendHPO` | `"optuna"` (unico valor soportado hoy) |

## Estados (`EstadoCorrida`)

`StrEnum` con valores JSON en minusculas:

| Miembro | Valor persistido |
| --- | --- |
| `NUEVA` | `nueva` |
| `EN_PROGRESO` | `en_progreso` |
| `INTERRUMPIDA` | `interrumpida` |
| `COMPLETADA` | `completada` |
| `FALLIDA` | `fallida` |

`ManifiestoCorrida.estado` usa `strict=False` en Pydantic para round-trip
JSON con enums.

## Contratos Pydantic (`contratos.py`)

Todos: `strict=True`, `extra=forbid`, `frozen=True` (salvo `estado` arriba).

### `ConfiguracionValidacion`

| Campo | Tipo |
| --- | --- |
| `min_train` | `int` (ge 1) |
| `horizonte` | `int` (ge 1) |
| `paso` | `int` (ge 1) |
| `estacionalidad` | `int` (ge 1) |

### `ConfiguracionOptimizador`

Recorte serializable de `ReglasPoda` + presupuesto + etiqueta de muestreador.

| Campo | Tipo |
| --- | --- |
| `n_trials_objetivo` | `int` (ge 1) |
| `muestreador` | `str` (nombre de clase, p. ej. `RandomSampler`) |
| `min_ventanas` | `int` (ge 1) |
| `agregacion` | `media` \| `mediana` \| `media_recortada` |
| `proporcion_recorte` | `float` [0, 0.5] |
| `factor_reduccion` | `int` (ge 2) |
| `habilitar_poda_semantica` | `bool` |

### `SolicitudCorrida`

Pedido de crear o reanudar. **`run_id` no entra en la huella.**

| Campo | Tipo | Default |
| --- | --- | --- |
| `run_id` | `str` (min 1, no solo espacios) | — |
| `familia` | `str` | — |
| `sku_id` | `str` | — |
| `seed` | `int` | — |
| `metrica_objetivo` | `str` | — |
| `validacion` | `ConfiguracionValidacion` | — |
| `espacio_busqueda` | `dict[str, Any]` | salida de `descripcion_canonica()` |
| `optimizador` | `ConfiguracionOptimizador` | — |
| `n_observaciones` | `int` (ge 1) | longitud de `y` |
| `huella_serie` | `str` | SHA-256 de `y` float64 |
| `backend` | `BackendHPO` | `"optuna"` |

### `ManifiestoCorrida`

Metadatos persistidos. Independiente de Optuna.

| Campo | Tipo |
| --- | --- |
| `schema_version` | `int` (ge 1) |
| `run_id` | `str` |
| `estado` | `EstadoCorrida` |
| `familia`, `sku_id`, `seed`, `metrica_objetivo` | como solicitud |
| `configuracion_validacion` | `ConfiguracionValidacion` |
| `fingerprint_configuracion` | `str` (huella SHA-256) |
| `backend` | `BackendHPO` |
| `backend_checkpoint` | `str \| None` |
| `n_trials_objetivo` | `int` |
| `n_trials_finalizados` | `int` (ge 0) |

## Huella (`huella.py`)

Campos que participan en `calcular_huella`:

`familia`, `sku_id`, `seed`, `metrica_objetivo`, `validacion`,
`espacio_busqueda`, `optimizador`, `n_observaciones`, `huella_serie`,
`backend`.

- Payload canonico → JSON ordenado → SHA-256 hex.
- `verificar_compatibilidad(solicitud, manifiesto)` lanza
  `IncompatibilidadCorridaError` con `motivos` explicitos.
- El rechazo por huella **no modifica** el manifiesto persistido.

## Transiciones (`transiciones.py`)

`transicionar(actual, nuevo) -> EstadoCorrida` o `TransicionEstadoError`.

Mapa completo en `TRANSICIONES_PERMITIDAS`. Incluye identidad en
`EN_PROGRESO` y `COMPLETADA`.

## Almacenamiento (`almacenamiento.py`)

### `escribir_atomico(ruta, contenido)`

Escribe `ruta.tmp` y `os.replace` al destino.

### `AlmacenManifiestosFs(raiz)`

Implementa `AlmacenManifiestos`:

| Metodo | Comportamiento |
| --- | --- |
| `guardar(manifiesto)` | JSON indentado, atomico, log INFO |
| `cargar(run_id)` | Valida `schema_version`, Pydantic |
| `existe(run_id)` | `manifiesto.json` es archivo |
| `ruta_de(run_id)` | `{raiz}/{run_id}/manifiesto.json` |

Errores: `ManifiestoAusenteError`, `ManifiestoCorruptoError`,
`VersionManifiestoError`.

## Controlador (`controlador.py`)

### `SesionCorrida`

`manifiesto: ManifiestoCorrida` + `handle: Any` (opaco del motor).

### `ControladorReanudacion(almacen, puerto)`

| Metodo | Efecto |
| --- | --- |
| `abrir(solicitud)` | Crea o restaura corrida; devuelve `SesionCorrida` |
| `checkpoint(sesion, n_trials_finalizados=...)` | Persiste backend + manifiesto `en_progreso` |
| `interrumpir(sesion, n_trials_finalizados=...)` | Persiste + estado `interrumpida` |
| `completar(sesion, n_trials_finalizados=...)` | Persiste + estado `completada` |
| `fallar(sesion, n_trials_finalizados=...)` | Persiste + estado `fallida` |

Comportamiento de `abrir`:

1. Sin manifiesto → `_crear`: huella, estado `nueva` → `en_progreso`,
   backend vacio.
2. `FALLIDA` → `CorridaFallidaError` (no reabre).
3. `COMPLETADA` + huella compatible → restaura handle, **no** transiciona a
   `en_progreso`, no reejecuta trials.
4. `INTERRUMPIDA` → log recuperacion senalizada → `en_progreso`.
5. `EN_PROGRESO` → log recuperacion no senalizada (kill -9).
6. Huella incompatible → `IncompatibilidadCorridaError` sin mutar manifiesto.

## Puerto del motor (`PuertoPersistenciaMotor`)

Protocol inyectable. `handle` opaco para el controlador.

| Metodo | Contrato |
| --- | --- |
| `crear()` | Nuevo estudio HPO |
| `persistir(handle)` | Serializa; retorna referencia (`backend.jsonl`) |
| `restaurar(referencia)` | Reconstruye handle desde JSONL |

## Adaptador Optuna (`AdaptadorPersistenciaOptuna`)

Unico implementador hoy. Vive en `adaptador_optuna.py`.

| Funcion / clase | Rol |
| --- | --- |
| `volcar_jsonl(study, ruta)` | Solo `COMPLETE` / `PRUNED` / `FAIL`; atomico |
| `reanudar_estudio(ruta, espacio, sampler, pruner)` | `add_trial` por linea JSONL; sincroniza `RandomSampler` |
| `_sincronizar_sampler_aleatorio` | Avanza RNG por `n_trials_cargados` |
| `persistir_estudio_hpo` / `restaurar_estudio_hpo` | Puente a `EstudioHPO` |

## Integracion HPO (`estudio.py`)

### Kwargs nuevos en `ejecutar_estudio`

| Parametro | Tipo | Default |
| --- | --- | --- |
| `controlador` | `ControladorReanudacion \| None` | `None` |
| `raiz_corrida` | `str \| Path \| None` | `None` |
| `run_id` | `str \| None` | `None` |

Reglas:

- `raiz_corrida` sin `run_id` → `EstudioError`.
- `controlador` sin `run_id` → `EstudioError`.
- `raiz_corrida` sin `controlador` → cablea `ControladorReanudacion`
  automaticamente via `_cablear_controlador`.
- Sin `raiz_corrida` → comportamiento previo (sin persistencia).

### Flujo con controlador

1. `abrir(SolicitudCorrida)` construida desde argumentos de
   `ejecutar_estudio`.
2. Si `COMPLETADA` → retorna `instantanea_desde_estudio` sin bucle.
3. Bucle `while indice_trial < n_trials`: `ask` → evaluar → `tell` →
   `checkpoint`.
4. Exito → `completar`.
5. `EstudioError` → `fallar`.
6. `Exception` / `KeyboardInterrupt` → `interrumpir` (best-effort).

### Helpers publicos

| Funcion | Rol |
| --- | --- |
| `huella_serie(y)` | SHA-256 de array 1D float64 |
| `_etiqueta_muestreador(muestreador)` | Nombre de clase o `"tpe"` |
| `_cablear_controlador(...)` | Ensambla almacen + adaptador Optuna |

## Excepciones (`errores.py`)

Jerarquia bajo `ReanudacionError`:

```
ReanudacionError
├─ TransicionEstadoError (ValueError)
├─ IncompatibilidadCorridaError  # .motivos: tuple[str, ...]
├─ CorridaFallidaError
├─ ManifiestoAusenteError
├─ ManifiestoCorruptoError
└─ VersionManifiestoError
```

`EstudioError` (HPO) es independiente; la integracion lo traduce a
`fallar` / propagacion segun el caso.

## Logging

`pred_engine.comun.logger.get_logger(__name__)`. Eventos: guardado /
lectura de manifiesto, checkpoint, interrupcion, recuperacion, rechazo
por huella, corrida completada. Sin PII ni series de demanda.
