# API — Evaluacion 3.5 (Persistencia y publicacion de evidencia)

Identificadores, comentarios y mensajes en espanol. Recibe una
`sqlite3.Connection` ya abierta sobre la base de la plataforma (con la
migracion 0002 aplicada); no crea tablas ni orquesta.

Punto de entrada: `pred_engine.forecasting.persistencia_evidencia`

## Constantes

| Nombre | Valor |
| --- | --- |
| `VENTANA_AGREGADA` | `"agregada"` (valor de `m3_metricas.ventana` para las metricas agregadas) |
| `TABLAS` | las ocho tablas `m3_*`; `publicar` las exporta todas |
| `TABLAS_POR_SKU` | las que tienen columna `sku`: `m3_skus`, `m3_fallos`, `m3_unidades`, `m3_pronosticos`, `m3_metricas`, `m3_veredictos` |

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
| `registrar_corrida` | no registrada | `EN_CURSO` con `m3_skus` (SKUs de candidatos y fallos) y `m3_fallos`, o `RECHAZADA` con `causa_rechazo = "motivo: detalle"`; si ya existe, no cambia nada |
| `guardar_unidades` | `EN_CURSO` | una unidad por origen y sus pronosticos (`h` desde 1); devuelve las nuevas; las ya confirmadas no se tocan. Sirve tambien para Seasonal Naive (`pronosticar_linea_base` de 3.4) |
| `marcar_evaluada` | `EN_CURSO`, `EVALUADA` | pasa a `EVALUADA` |
| `guardar_evaluacion` | `EVALUADA`, `SELECCIONADA` | controla versiones, origen de datos y pronosticos confirmados; escribe metricas, categorias y veredictos (con Diebold-Mariano) y pasa a `SELECCIONADA`; en `SELECCIONADA` no escribe |
| `publicar` | `SELECCIONADA`, `PUBLICADA` | exige veredicto para todo SKU de `m3_skus`, pasa a `PUBLICADA` y escribe `directorio/run_id/<tabla>.parquet` en la misma transaccion; en `PUBLICADA` solo devuelve las rutas |

- Fechas como texto `AAAA-MM-DD`. Valores no finitos o de forma invalida se
  guardan como NULL.
- `m3_metricas` guarda `n_pares`, `mae`, `rmse`, `me`, `mase` y `razon_sn` por
  ventana y agregadas, mas `n_ventanas_totales`, `n_ventanas_validas` y
  `n_pronosticos_validos` en `agregada`.
- `m3_categorias` guarda como JSON `medianas_r`, `skus_comparables`,
  `excluidos`, `familias_excluidas`, `conteos` y `porcentajes`.
- `m3_veredictos` guarda la prueba de `VeredictoSku.diebold_mariano` en
  `dm_n_ventanas`, `dm_horizonte`, `dm_estadistico`, `dm_p_valor`,
  `dm_p_ajustado`, `dm_alfa`, `dm_significativa` y `dm_causa`.
- `consultar` lee en orden de insercion, con el tipo declarado en el esquema
  para cada columna (`INTEGER` -> `Int64`, `REAL` -> `Float64`, `TEXT` ->
  `string`); con `sku` devuelve solo `TABLAS_POR_SKU`. El Parquet es
  exactamente ese `DataFrame`.

## Vista `evidencia_publicada`

`m3_veredictos.*` mas `datos_sinteticos` e `identidad` de la corrida, solo para
corridas `PUBLICADA`. Es lo que consume M4 (ADR-03-010).

## Excepciones (`errores.py`)

```
PersistenciaEvidenciaError(Exception)
├── EstadoCorridaError(run_id, estado, operacion)   # no registrada, fase equivocada o cerrada
└── EvidenciaInconsistenteError                     # versiones u origen de datos distintos,
                                                    # modelo sin pronosticos, SKU sin veredicto
```

Los errores de SQLite (por ejemplo, `ingesta_sha256` que no esta en
`ingestas`, evidencia de un SKU fuera de `m3_skus`, o un `CHECK` violado) se
propagan como `sqlite3.IntegrityError` tras revertir la transaccion. Un error
al escribir el Parquet tambien revierte la publicacion.

## Logging

- `INFO` al registrar (SKUs y fallos), por lote de unidades (nuevas de total)
  y en cada cambio de estado.
- `ERROR` antes de cada `EstadoCorridaError` o `EvidenciaInconsistenteError`
  (`operacion`, `run_id`, `estado`) y al revertir por cualquier otro error.
- Sin series de demanda ni PII en los mensajes.
