# API — Evaluacion 3.5 (Persistencia y publicacion de evidencia)

Identificadores, comentarios y mensajes en espanol. Recibe una
`sqlite3.Connection` ya abierta sobre la base de la plataforma (con la
migracion 0002 aplicada); no crea tablas ni orquesta.

Punto de entrada: `pred_engine.forecasting.persistencia_evidencia`

## Constantes

| Nombre | Valor |
| --- | --- |
| `VENTANA_AGREGADA` | `"agregada"` (valor de `m3_metricas.ventana` para las metricas agregadas) |
| `TABLAS` | las siete tablas `m3_*` |
| `TABLAS_POR_SKU` | las que tienen columna `sku`: `m3_fallos`, `m3_unidades`, `m3_pronosticos`, `m3_metricas`, `m3_veredictos` |
| `TABLAS_EXPORTADAS` | `TABLAS` sin `m3_corridas` |

## Tipos (`contratos.py`)

### `EstadoCorrida`

`Literal["EN_CURSO", "EVALUADA", "SELECCIONADA", "PUBLICADA", "RECHAZADA"]`.

### `IdentidadCorrida` (Pydantic, estricto y congelado)

| Campo | Tipo |
| --- | --- |
| `ingesta_ref_m1` | `str` (hash del Parquet de M1) |
| `configuracion_id` | `str` |
| `codigo_version` | `str` |
| `t_corte_reserva` | `date` |
| `version_politica` | `str` |
| `version_metricas` | `str` |

- `canonica() -> str`: JSON con claves ordenadas; es lo que se guarda en
  `m3_corridas.identidad`.
- `run_id -> str`: `"m3-"` + 16 hex del SHA-256 de `canonica()`.

## Funciones (`repositorio.py`)

```python
esquema_m3() -> str
registrar_corrida(
    conn, identidad: IdentidadCorrida,
    handoff: HandoffValidado | LoteRechazadoError,
    *, ingesta_sha256: str, datos_sinteticos: bool,
) -> str
guardar_unidades(conn, run_id: str, sku: str, serie: SerieCandidato) -> int
unidades_confirmadas(conn, run_id: str) -> frozenset[tuple[str, str, pd.Timestamp]]
marcar_evaluada(conn, run_id: str) -> None
guardar_evaluacion(
    conn, run_id: str,
    evaluaciones: Sequence[EvaluacionSku], resultado: ResultadoEvaluacion,
) -> None
publicar(conn, run_id: str, directorio: str | Path) -> dict[str, Path]
estado_corrida(conn, run_id: str) -> EstadoCorrida | None
consultar(conn, run_id: str, *, sku: str | None = None) -> dict[str, pd.DataFrame]
```

| Funcion | Estados admitidos | Efecto |
| --- | --- | --- |
| `registrar_corrida` | no registrada | `EN_CURSO` con `m3_fallos`, o `RECHAZADA` con `causa_rechazo = "motivo: detalle"`; si ya existe, no cambia nada |
| `guardar_unidades` | `EN_CURSO` | una unidad por origen y sus pronosticos (`h` desde 1); devuelve las nuevas; las ya confirmadas no se tocan |
| `marcar_evaluada` | `EN_CURSO`, `EVALUADA` | pasa a `EVALUADA` |
| `guardar_evaluacion` | `EVALUADA`, `SELECCIONADA` | metricas, categorias y veredictos, y pasa a `SELECCIONADA`; en `SELECCIONADA` no escribe |
| `publicar` | `SELECCIONADA`, `PUBLICADA` | verifica veredictos, escribe `directorio/run_id/<tabla>.parquet` y pasa a `PUBLICADA`; en `PUBLICADA` solo devuelve las rutas |

- Fechas como texto `AAAA-MM-DD`. Valores no finitos o de forma invalida se
  guardan como NULL.
- `m3_metricas` guarda `n_pares`, `mae`, `rmse`, `me`, `mase` y `razon_sn` por
  ventana y agregadas, mas `n_ventanas_totales`, `n_ventanas_validas` y
  `n_pronosticos_validos` en `agregada`.
- `consultar` lee con `dtype_backend="numpy_nullable"`, en orden de insercion;
  con `sku` devuelve solo `TABLAS_POR_SKU`. El Parquet es exactamente ese
  `DataFrame`.

## Excepciones (`errores.py`)

```
PersistenciaEvidenciaError(Exception)
├── EstadoCorridaError(run_id, estado, operacion)   # no registrada, fase equivocada o cerrada
└── EvidenciaInconsistenteError                     # versiones distintas a la identidad, SKU sin veredicto
```

Los errores de SQLite (por ejemplo, `ingesta_sha256` que no esta en
`ingestas`, o un `CHECK` violado) se propagan como `sqlite3.IntegrityError`
tras revertir la transaccion.

## Logging

- `INFO` al registrar, por lote de unidades (nuevas de total) y en cada cambio
  de estado.
- `ERROR` antes de cada `EstadoCorridaError` o `EvidenciaInconsistenteError`
  (`operacion`, `run_id`, `estado`) y al revertir por un error de SQLite.
- Sin series de demanda ni PII en los mensajes.
