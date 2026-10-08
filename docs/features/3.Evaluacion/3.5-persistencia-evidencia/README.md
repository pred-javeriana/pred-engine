# Evaluacion 3.5 — Persistencia y publicacion de evidencia

## Que se hizo

Implementacion de la **persistencia estructurada de la evidencia de M3**
(`TASK-EVAL-3.5-A1`), segun ADR-03-009 y ADR-03-010.

1. **Esquema SQLite** (`esquema_m3.sql`): ocho tablas `m3_*` con claves
   naturales y la vista `evidencia_publicada`. Es la fuente unica: la
   plataforma lo aplica tal cual como su migracion `0002_evidencia_m3.sql`.
2. **Identidad de corrida** (`contratos.py`): `IdentidadCorrida` con los
   campos de 3.0-A1 mas las versiones de politica y metricas; su JSON canonico
   da el `run_id` (`m3-` + 16 hex).
3. **Repositorio** (`repositorio.py`): registro de la corrida con el resultado
   de 3.2 y los SKUs de su manifiesto, unidades confirmadas con sus
   pronosticos (incluidos los de Seasonal Naive), metricas, seleccion,
   veredictos y prueba Diebold-Mariano de 3.4, maquina de estados y consultas
   por corrida y por SKU.
4. **Publicacion**: `publicar` pasa la corrida a `PUBLICADA` y exporta el
   Parquet derivado de SQLite en la misma transaccion.

## Como

```
3.2  validar_manifiesto ─► HandoffValidado | LoteRechazadoError
                             │ registrar_corrida
                             ▼
        EN_CURSO (SKUs del manifiesto, fallos)  o  RECHAZADA (causa) ── final
                             │ guardar_unidades(sku, SerieCandidato)
                             │   por candidato y por Seasonal Naive (3.3 y 3.4);
                             │   una unidad por origen: marcador + pronosticos, misma transaccion
                             │ marcar_evaluada
                             ▼
        EVALUADA             │ guardar_evaluacion(EvaluacionSku[], ResultadoEvaluacion)  (3.4)
                             ▼   metricas + categorias + veredictos, misma transaccion
        SELECCIONADA         │ publicar(directorio)
                             ▼   verifica veredictos, pasa a PUBLICADA y exporta Parquet,
                                 misma transaccion
        PUBLICADA ── final; M4 lee la vista evidencia_publicada
```

- **Idempotencia.** Registrar otra vez la misma identidad no cambia nada.
  Una unidad ya confirmada no se reescribe, aunque llegue con otros valores.
  Una fase ya cumplida no vuelve a escribir.
- **Atomicidad.** Cada escritura abre `BEGIN IMMEDIATE`: si algo falla a
  mitad, no queda ni marcador ni pronosticos. Antes ejecuta
  `PRAGMA foreign_keys = ON`, que SQLite trae apagado.
- **Reanudacion.** `unidades_confirmadas` devuelve `(sku, candidato_id,
  origen)` para que quien coordine (3.0) no recalcule. `estado_corrida` dice
  desde que fase seguir.
- **Corridas cerradas.** Sobre `PUBLICADA` o `RECHAZADA` no se escribe
  (`EstadoCorridaError`). Re-evaluar exige otra identidad, es decir, otro
  `run_id`.

### Controles antes de escribir la evaluacion y de publicar

`guardar_evaluacion` rechaza con `EvidenciaInconsistenteError`, sin escribir:

- versiones de politica o metricas distintas de las de la identidad;
- veredictos emitidos con otro origen de datos que el de la corrida
  (`ResultadoEvaluacion.datos_sinteticos`): con datos sinteticos no puede
  quedar un `VALIDADO` ni una prueba Diebold-Mariano (ADR-03-008);
- un modelo evaluado con ventanas, incluida la linea base, sin pronosticos
  confirmados en `m3_unidades`.

`publicar` exige veredicto para todo SKU de `m3_skus`, es decir, del manifiesto.

### Tablas

| Tabla | Clave natural | Contenido |
| --- | --- | --- |
| `m3_corridas` | `run_id` | estado, identidad (JSON), `ingesta_sha256` (FK a `ingestas`), `run_id_m2`, `datos_sinteticos`, causa de rechazo |
| `m3_skus` | `run_id, sku` | SKUs del manifiesto validado (candidatos y fallos de 3.2) |
| `m3_fallos` | `run_id, candidato_id` | candidatos que 3.2 no valido |
| `m3_unidades` | `run_id, sku, candidato_id, origen` | marcador de unidad confirmada, con familia y modelo |
| `m3_pronosticos` | `run_id, sku, candidato_id, origen, h` | fecha y valor (NULL si no fue finito) |
| `m3_metricas` | `run_id, sku, candidato_id, ventana, metrica` | por ventana y `agregada`; NULL con su `causa` si no es calculable |
| `m3_categorias` | `run_id, sku_class` | familia campeona, motivo, medianas de r, SKUs excluidos, familias excluidas, conteos |
| `m3_veredictos` | `run_id, sku` | veredicto ADR-03-008, campeon, r, cobertura, IQR, prueba Diebold-Mariano (`dm_*`), `modelos_evaluados` |

`m3_fallos`, `m3_unidades`, `m3_metricas` y `m3_veredictos` referencian
`m3_skus`: no se guarda evidencia de un SKU que no esta en el manifiesto.

### Diebold-Mariano para M4

`m3_veredictos` guarda la prueba HLN-DM de 3.4 en `dm_n_ventanas`,
`dm_horizonte`, `dm_estadistico`, `dm_p_valor`, `dm_p_ajustado` (BH entre los
SKUs de la corrida), `dm_alfa`, `dm_significativa` y `dm_causa`. Un `CHECK`
exige que cada fila tenga p-valor o causa, nunca ambos ni ninguno. M4 la lee en
`evidencia_publicada` y en el Parquet de `m3_veredictos`.

La vista `evidencia_publicada` une veredictos y corrida solo para corridas
`PUBLICADA`; las demas tablas se filtran por los `run_id` que aparecen en ella.

## Donde

| Pieza | Ruta |
| --- | --- |
| Esquema, identidad y repositorio | `src/pred_engine/forecasting/persistencia_evidencia/` |
| Migracion de la plataforma | `pred-platform`, `src/pred_platform/dal/migrations/0002_evidencia_m3.sql` (copia exacta del esquema) |
| Pruebas | `tests/forecasting/persistencia_evidencia/` |
| Constructores de prueba | `tests/_evidencia.py` |
| API | `API_SPECIFICATION.md` |
| Notion | ADR-03-009, ADR-03-010; TASK-EVAL-3.5-A1 |

## Mejoras respecto al texto de Notion (y por que)

- **Tablas nuevas `m3_*` mediante migracion, no las tablas existentes.**
  ADR-03-009 pide usar las tablas de `pred-platform` y agregar lo que falte
  con una migracion versionada (ADR-05-002). Las de la version 1 no sirven
  como estan: `reportes_validacion` solo admite `mantiene/parcial/falla`
  (no los cinco veredictos de ADR-03-008); ni ella ni `pronosticos` tienen
  clave unica, y sin clave unica no hay `ON CONFLICT`. `metricas.valor` es
  `NOT NULL` y `resultados_comparativos` no admite Seasonal Naive como
  campeon. Cambiarlas romperia el contrato v1 de la plataforma.
- **Un solo texto para el esquema.** `pred-platform` depende de `pred-engine`,
  no al reves. Por eso el DDL vive aqui (`esquema_m3()`) y la migracion 0002
  debe ser una copia exacta, con su SHA-256 fijado en las pruebas de la
  plataforma. El motor nunca crea tablas en la base de la plataforma. Un
  cambio al esquema despues de publicar la 0002 exige una migracion nueva.
- **Dos referencias a la ingesta.** `ingestas.sha256` es el hash del CSV
  fuente; `ingesta_ref_m1` en el manifiesto es el hash del Parquet publicado.
  La FK usa el primero (`ingesta_sha256`) y la identidad el segundo.
- **Los SKUs del manifiesto se guardan** (`m3_skus`) para que "todo SKU del
  manifiesto tiene veredicto" (ADR-03-010) sea exacto, incluso para un SKU sin
  unidades.
- **La unidad confirmada es la fila de `m3_unidades`.** Un pronostico no
  puede existir sin ella (FK), asi que "no quedan unidades sin marcador" se
  cumple por construccion.
- **Las unidades se guardan por candidato** (`SerieCandidato`, lo que recibe
  3.4): cada origen sigue siendo su propia unidad, agrupadas en una
  transaccion, como permite ADR-03-010. Seasonal Naive se guarda igual: sus
  pronosticos tambien son evidencia.
- **Sin estado `fallida` por unidad.** 3.3 aun no define como falla una
  ventana; hoy un pronostico no finito se guarda con valor NULL y 3.4 lo
  cuenta como ventana invalida. Se agrega cuando 3.3 lo defina.
- **Parquet.** Se mantiene porque el criterio de 3.5-A1 lo exige. Se exportan
  las ocho tablas; el estado cambia antes de exportar, dentro de la
  transaccion, asi el Parquet de `m3_corridas` ya dice `PUBLICADA` y, si la
  exportacion falla, la corrida sigue en `SELECCIONADA`. Cada columna conserva
  el tipo declarado en el esquema aunque venga toda en NULL. Su consumidor
  debe confirmarse con 3.5-A2 (Derek); si nadie lo lee, se elimina
  (ADR-03-009).
- **`datos_sinteticos`** se guarda por corrida, la vista lo expone a M4 y se
  contrasta con el de los veredictos.
- **Las metricas del campeon y de SN** no se duplican en `m3_veredictos`:
  estan en `m3_metricas` (ventana `agregada`) de su `candidato_id`.

## Verificacion

```bash
uv run pytest tests/forecasting/persistencia_evidencia -q
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest -m "not slow"
```

- `pyproject.toml`: pyright incluye `forecasting/persistencia_evidencia`.
- La copia del esquema en la plataforma la verifican las pruebas de
  migraciones de `pred-platform`.

## Fuera de alcance y pendientes

- Orquestar M3 y la reanudacion por fase (3.0 y L4, Derek): aqui solo se
  exponen el estado y las unidades confirmadas.
- Generar las ventanas de la reserva (3.3, Derek); el progreso de una corrida
  `EN_CURSO` (unidades confirmadas sobre el total) y las unidades fallidas con
  causa esperan su contrato.
- `run_id_m2` se guarda sin FK: las tablas `ejecuciones` y `tareas` de la
  plataforma siguen vacias hasta que exista la orquestacion.
- El contrato de consumo de M4 y la exportacion a CSV (3.5-A2, Derek).
- Actualizar el contrato v1 de la plataforma (`validation_verdicts` lee
  `reportes_validacion`, con otro vocabulario de veredictos y sin
  Diebold-Mariano).
