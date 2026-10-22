# Módulo 3: evaluación retrospectiva y selección definitiva

M3 recibe el Parquet inmutable de M1 y los candidatos completos de M2. Evalúa
cada configuración sobre todas las ventanas completas de la reserva final del
20 %, compara con Seasonal Naive, selecciona por categoría y publica los
resultados de cada SKU entregado en el handoff y de cada SKU cuyos ajustes L2
fallaron todos. Estos últimos reciben FALLO_TECNICO con la causa registrada
por L2 y cuentan en los conteos y porcentajes de su clase. Sus unidades de
línea base se incluyen en el total previsto. Las exclusiones previas de M2
se consultan por separado en la procedencia. M3 no ejecuta HPO, no retunea y
no desarrolla la interfaz de M4.

## Ejecución

La base central debe existir. La plataforma la crea mediante su migrador:

```bash
python -m pred_platform.dal.migrate --db data/pred.db
pred-engine run --csv ventas.csv --data-root data --db-path data/pred.db \
  --families classical ml --horizon 7 --step 28 --workers 4
```

El motor aplica automáticamente las migraciones M3 pendientes. Una base
inexistente produce un error claro antes de ejecutar M0 o M2. No crea otra
base. `--db-path` tiene prioridad sobre `PRED_DB_PATH`; sin ambos se usa
`{data-root}/pred.db`. La plataforma pasa su ruta efectiva al proceso del motor.

Para datos aumentados o sintéticos se debe indicar `--datos-sinteticos`.
`--seed-csv` activa esa condición automáticamente. El protocolo conserva el
origen de los datos y rechaza cambiarlo para una misma ingesta. M1 publica
`source_file.datos_sinteticos` y `published.sha256`; la importación conserva
esa procedencia para el proceso que consume el Parquet. Un Parquet externo
sin procedencia mantiene origen desconocido. Las opciones `--datos-sinteticos`
y `--datos-reales` declaran explícitamente su origen; nunca se presume
real por omisión. El origen desconocido produce resultados exploratorios sin
valores p; no habilita VALIDADO. Se reutiliza una declaración anterior por
huella CSV o Parquet y un conflicto se rechaza antes de M1/HPO.
`ControladorEvaluacion` aplica la misma regla en la biblioteca: resuelve el
origen registrado en la base o en el informe M1, el origen declarado en
`PipelineInput` y la declaración del validador, y rechaza un conflicto antes
de registrar la ejecución. Una ruta Parquet registrada cuyo contenido cambió
ya no identifica su ingesta: se rechaza en lugar de heredar su procedencia.
La ejecución central guarda los ajustes efectivos de M2 que `build_pipeline`
usó (familias, ensayos, semilla); un `Pipeline` compuesto a mano los declara
con `m2_settings`. Un `run_id` de M2 ya registrado en la base solo se
reutiliza con los mismos ajustes que cambian resultados (familias, ensayos,
`min_train`, horizonte, paso, estacionalidad, métrica, semilla, y la versión
de código cuando ambas partes la registran); `workers` puede cambiar.
El origen declarado se asocia también a la huella de los bytes Parquet: otra
ingesta que publica los mismos bytes no puede declarar el origen contrario, y
una ruta registrada sin ninguna huella conocida no aporta procedencia. M3
registra la huella de los bytes que L1 leyó, no la del archivo al llegar a
L4; un `IngestionArtifact` construido a mano sin `parquet_sha256` se rechaza
(use `Pipeline.ingest` o `run_verify_parquet_with_sha256`). El protocolo congelado guarda la configuración M2 efectiva, incluida la
versión de código del validador (`code`); una
`configuracion_m2` del llamador que la contradiga se rechaza antes de
registrar la ejecución.

- Salida 0: etapas completas y evidencia PUBLICADA.
- Salida 8: publicación completa con fallos técnicos aislados en M2 o M3.
- Salida 1: entrada, identidad, base o etapa inválida.
- Un pipeline de biblioteca sin validador configurado puede conservar L4
  bloqueada. No representa una evaluación realizada.

Los días reservados sin observación producen exclusiones de datos, no fallos
técnicos. Se omiten los orígenes sin contexto y se identifican las ventanas
sin objetivos completos. No se imputan ceros ni se acorta H. La ejecución de
plataforma solo termina después del resultado L4 y conserva su error si falla.

## Contrato de biblioteca

```python
from pred_engine.forecasting.control_reanudacion import ControladorEvaluacion
from pred_engine.forecasting.persistencia_evidencia import abrir_base_plataforma
from pred_engine.pipeline import EvaluationSettings, PipelineInput
from pred_engine.pipeline_setup import build_pipeline

with abrir_base_plataforma("data/pred.db") as conn:
    validator = ControladorEvaluacion(
        conn, "data/runs/mi-corrida",
        run_id_m2="mi-corrida", codigo_version="revision-del-codigo",
        datos_sinteticos=False, data_root="data",
    )
    pipeline = build_pipeline(
        EvaluationSettings(min_train=60, horizon=7, step=28),
        families=("classical", "ml"), validator=validator,
    )
    result = pipeline.run(PipelineInput(csv_path="ventas.csv", data_root="data", datos_sinteticos=False))
```

La conexión pertenece al llamador. La configuración y el identificador de la
corrida deben ser estables al reanudar. El CLI construye esa identidad desde
la entrada, las opciones y la versión de código. Guarda `identidad.json`
antes del HPO, por separado del resumen de cada intento. Rechazar otra entrada
no sustituye esa identidad ni bloquea la reanudación correcta. Si una corrida
antigua perdió tanto la identidad como un resumen completo, se exige restaurar
esa identidad o usar otra corrida; los checkpoints parciales no prueban por sí
solos cuál era el archivo de entrada original. SQLite conserva
el manifiesto autorizado y se consulta antes de restaurar un archivo perdido.

## Capacidades y propietarios

| Sección | Capacidad | Código | Pruebas |
| --- | --- | --- | --- |
| 3.0 | Identidad, fases y recuperación durable | `forecasting/control_reanudacion/`, `persistencia_evidencia/recuperacion.py` | `tests/forecasting/test_control_evaluacion.py`, pruebas de persistencia |
| 3.1 | Reserva fija, exclusiones y causalidad | `comun/reserva.py`, `generador_pronosticos/particion.py` | `tests/test_pipeline.py`, `test_generacion_reserva.py` |
| 3.2 | Manifiesto estricto y fábricas compartidas | `forecasting/adaptador_candidatos/` | `tests/forecasting/adaptador_candidatos/` |
| 3.3 | Ajuste único y ventanas diarias completas | `forecasting/generador_pronosticos/` | `tests/forecasting/test_generacion_reserva.py` |
| 3.4 | Métricas, selección, cinco veredictos y HLN-DM/BH | `forecasting/evaluaciones/` | `tests/forecasting/evaluaciones/` |
| 3.5 | SQLite, Parquet/CSV y cartera precalculada | `forecasting/persistencia_evidencia/` | `tests/forecasting/persistencia_evidencia/`, prueba de control |
| 3.6 | Lectura del Parquet M1 sin cambiarlo | `Pipeline.ingest`, `ReserveCut`, generador | pruebas del pipeline y de control con entrada M1 |
| 3.7 | Trazabilidad, límites y auditoría | este documento, ADR-021 y documentos de cada capacidad | suite completa y corrida coordinada de integración |

Las rutas de código son relativas a `src/pred_engine/`. La plataforma consume
la evidencia mediante `pred_platform.dal.evidencia.read_m3_evidence`, sin
recalcular métricas. Las carpetas de `analisis/` pertenecen a M4.

## Recuperación

```text
handoff inválido -> RECHAZADA
handoff válido  -> EN_CURSO
                   -> EVALUADA
                   -> SELECCIONADA
                   -> PUBLICADA
```

EN_CURSO solo ejecuta orígenes pendientes. Una falla conserva fechas, valores
NULL y causa, nunca ceros imputados. EVALUADA permite calcular métricas y
selección con los pronósticos durables. SELECCIONADA solo debe publicar.
PUBLICADA y RECHAZADA son terminales. Cambiar configuración exige otra
identidad. La confirmación de resultado y marcador es atómica.

## Evidencia científica

Se publican MAE, RMSE, MASE, ME y la razón de RMSE frente a Seasonal Naive por
candidato y ventana. La política `3.4.2-inicial` selecciona por mediana de la
razón dentro de cada categoría, con compuerta de línea base y desempate por
simplicidad. Los cinco veredictos permanecen visibles por SKU.

HLN-DM con BH acompaña esas métricas. `--dm-min-origins` fija el mínimo de
orígenes comunes antes de abrir la reserva; el valor inicial es 30. La prueba
solo usa datos reales y H coincide con el horizonte de cada ventana. Una
muestra insuficiente produce una causa sin p-valor. La prueba no cambia el
veredicto descriptivo y los datos sintéticos nunca se prueban.

La selección y el puntaje comparten reserva. Por tanto, los resultados no
constituyen una prueba no sesgada de desempeño futuro. ADR-021 documenta los
límites y las contradicciones de los textos anteriores.

## Aceptación del hito

La revisión técnica debe comprobar la suite completa con cobertura >= 80 %,
la compatibilidad de las migraciones de ambos paquetes y una corrida conjunta
con base central temporal, múltiples SKU, reanudación, actualización de una
versión pendiente y rechazo de una base ausente. También debe medir tiempo,
CPU, memoria y costo de reinicio en el equipo efectivo. No se usa VRAM en el
catálogo CPU de este hito. La etapa L4 corre en serie en el proceso
principal, con BLAS de varias hebras; `--workers` solo paraleliza L2 y L3, así
que el tiempo de L4 crece con el número de SKU.

La firma del líder de ingeniería, la aprobación de otro fundador, la fusión y el cierre de
las tareas del Hub son actos posteriores de aceptación. Este documento no
sustituye esas autorizaciones.
