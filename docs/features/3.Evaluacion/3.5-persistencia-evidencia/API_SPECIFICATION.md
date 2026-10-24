# API: persistencia de evidencia M3

Punto de entrada: `pred_engine.forecasting.persistencia_evidencia`.
La conexión pertenece al llamador. Cada operación abre su propia transacción;
no se debe entregar una conexión con trabajo ajeno pendiente.

## Base central y migraciones

```python
abrir_base_plataforma(ruta: str | Path)  # context manager -> sqlite3.Connection
aplicar_migraciones_m3(conn) -> tuple[int, ...]
textos_migraciones_m3() -> tuple[tuple[int, str], ...]
esquema_m3() -> str
```

`abrir_base_plataforma` usa modo `rw`, nunca crea un archivo ausente y aplica
solo las migraciones M3 pendientes. La versión vigente es 9, con las migraciones 0005 a 0009. Se exige una base
de plataforma de versión 4 o posterior conocida. Base ausente, transacción
ajena o versión desconocida producen `BasePlataformaError` sin crear otra base.
La migración y el cambio de versión son atómicos; una repetición no los duplica.

## Identidad y registro

`IdentidadCorrida` es Pydantic estricto e inmutable: `ingesta_ref_m1`,
`configuracion_id`, `codigo_version`, `t_corte_reserva`, `version_politica` y
`version_metricas`. `canonica()` produce JSON ordenado y `run_id` lo identifica
como `m3-` seguido de 16 dígitos hexadecimales.

```python
registrar_corrida(
    conn, identidad: IdentidadCorrida,
    handoff: HandoffValidado | LoteRechazadoError,
    *, ingesta_sha256: str, datos_sinteticos: bool | None, run_id_m2: str | None = None,
) -> str
```

Un handoff válido crea EN_CURSO, sus SKU y sus fallos de candidato. Un rechazo
crea RECHAZADA con causa o marca así la corrida incompleta existente. Una
PUBLICADA conserva su estado y evidencia. Repetir un handoff compatible no
cambia los datos.
Referencias u origen de datos incompatibles producen
`EvidenciaInconsistenteError`. El origen real o sintético queda ligado a la
ingesta, no se cambia para obtener otro resultado estadístico. `None` es origen
desconocido: no admite VALIDADO ni valores p. En SQLite, `origen_declarado=0`
representa esa ausencia; el bit `datos_sinteticos=0` por sí solo no declara
datos reales. Las declaraciones conocidas tienen `origen_declarado=1`.

## Unidades

```python
guardar_unidades(
    conn, run_id: str, sku: str, serie: SerieCandidato,
    *, codigo_version: str | None = None, semilla: int | None = None,
    emitido_en: datetime | None = None,
    fallos: Mapping[pd.Timestamp, FalloPronostico] | None = None,
) -> int
unidades_confirmadas(conn, run_id) -> frozenset[tuple[str, str, pd.Timestamp]]
marcar_evaluada(conn, run_id) -> None
```

Cada origen confirma marcador y sus H valores en la misma transacción. La
función devuelve las unidades nuevas. Una unidad ya confirmada o fallida no
se reescribe. Los fallos conservan estado `fallida` y causa estructurada; los
valores inválidos se guardan como NULL. Si hay protocolo congelado, no se
pasa a EVALUADA hasta confirmar el total previsto.

## Recuperación

Punto de entrada: `pred_engine.forecasting.persistencia_evidencia.recuperacion`.

```python
fijar_protocolo(
    conn, run_id, *, protocolo, manifiesto, procedencia,
    reserva_abierta_en, n_unidades_previstas, ejecucion_id=None,
) -> None
guardar_unidades_generadas(conn, run_id, unidades: Sequence[UnidadPronostico]) -> int
cargar_unidades(conn, run_id, candidato) -> tuple[UnidadPronostico, ...]
cargar_serie(conn, run_id, sku, candidato_id, familia, modelo) -> SerieCandidato
candidatos_con_fallos(conn, run_id, sku) -> frozenset[str]
protocolo_guardado(conn, run_id) -> dict | None
```

`guardar_unidades_generadas` confirma varias ventanas, con sus marcadores, en
una sola transacción y devuelve cuántas eran nuevas. El protocolo, el handoff
y el plan no se reemplazan. Las lecturas reconstruyen
fechas, NaN para NULL y metadatos originales. Una secuencia H incompleta,
fechas corruptas o trazabilidad incompatible se rechazan antes de reutilizar
la unidad. Los candidatos con fallos parciales también limitan la comparación
mediante `comparacion_incompleta`.

## Evaluación y publicación

```python
guardar_evaluacion(conn, run_id, evaluaciones, resultado) -> None
publicar(conn, run_id, directorio: str | Path) -> dict[str, Path]
estado_corrida(conn, run_id) -> EstadoCorrida | None
consultar(conn, run_id, *, sku: str | None = None) -> dict[str, pd.DataFrame]
```

| Operación | Estado admitido | Resultado |
| --- | --- | --- |
| `guardar_unidades` | EN_CURSO | Unidades y pronósticos nuevos |
| `marcar_evaluada` | EN_CURSO, EVALUADA | EVALUADA |
| `guardar_evaluacion` | EVALUADA, SELECCIONADA | Métricas, selección, veredictos y cartera; SELECCIONADA |
| `publicar` | SELECCIONADA, PUBLICADA | Exportaciones completas y PUBLICADA |

`guardar_evaluacion` comprueba versiones, origen de datos y ventanas durables
antes de escribir. Publica MAE, RMSE, MASE, ME y razón por candidato, ventana
y agregadas. `m3_cartera` contiene media, mediana, percentiles 25 y 75 y
conteos de SKU, instancias y valores no calculables por clase/familia/métrica.

`publicar` devuelve las rutas Parquet por tabla. Cada ruta tiene su CSV
hermano con extensión `.csv`. Ambos se derivan de SQLite dentro de la
publicación. Los archivos se preparan antes del renombrado del directorio y
la vista solo expone la corrida después del commit. Una corrida PUBLICADA no
se modifica; otra evaluación exige identidad nueva.

`consultar` conserva los tipos SQLite: INTEGER -> `Int64`, REAL -> `Float64`,
TEXT -> `string`. Con `sku` devuelve solo tablas con esa columna. Los NULL
conservan su causa en las métricas y fallos. CSV no codifica tipos por sí solo;
SQLite y Parquet siguen siendo la referencia tipada.

## Lectura de plataforma

`pred_platform.dal.evidencia.read_m3_evidence(db_path, run_id, sku=None)` acepta
la identidad M2 o M3 y devuelve registros precalculados solo de corridas
PUBLICADA. `m3_run_status` devuelve estado, causa y avance sin métricas
parciales. Ambas funciones abren SQLite en modo de solo lectura.

`EstadoCorridaError` rechaza escrituras en una fase incorrecta o cerrada.
`EvidenciaInconsistenteError` rechaza incoherencias de datos. Un error SQLite
o de exportación se propaga tras revertir la transacción. Los logs contienen
operación, corrida, SKU y causa, no series completas de demanda.
