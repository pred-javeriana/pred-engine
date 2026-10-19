# Evaluación 3.5: persistencia y publicación de evidencia

M3 conserva sus resultados en la base SQLite única de la plataforma.
`forecasting.persistencia_evidencia` recibe una conexión abierta y ofrece
escrituras transaccionales, consultas por corrida y SKU y exportaciones
derivadas en Parquet y CSV. No crea una base alternativa.

## Esquema y migraciones

El motor conserva los textos canónicos:

- `esquema_m3.sql`: esquema inicial, inmutable.
- `esquema_m3_0002.sql`: protocolo congelado, fallos durables por ventana,
  procedencia y resúmenes de cartera.

La plataforma incorpora copias exactas como migraciones 0005 y 0006.
`aplicar_migraciones_m3` usa el mismo `PRAGMA user_version`. El CLI aplica las
pendientes sobre una base existente con las migraciones previas de plataforma.
No modifica las versiones 0001 a 0004 y rechaza una versión desconocida.
El comentario histórico del esquema inicial menciona 0002; la numeración
vigente y su motivo se documentan en ADR-021.

## Evidencia durable

| Tabla | Contenido |
| --- | --- |
| `m3_corridas` | Identidad, referencia de ingesta fuente, corrida M2, origen de datos, estado y rechazo |
| `m3_protocolos` | Protocolo y handoff congelados, apertura, procedencia, ejecución de plataforma y unidades previstas |
| `m3_skus` | SKU del handoff, incluidos los que solo tienen fallos de adaptación |
| `m3_fallos` | Fallos del adaptador con candidato, campo y causa |
| `m3_unidades` | Marcador por candidato y origen, estado, causa, emisión, código y semilla |
| `m3_pronosticos` | Fechas y valores por origen y paso; NULL cuando el pronóstico falló |
| `m3_metricas` | MAE, RMSE, MASE, ME, razón y cobertura, por ventana y agregadas |
| `m3_categorias` | Selección, exclusiones, medianas y conteos de veredictos |
| `m3_cartera` | Distribuciones precalculadas por clase, familia y métrica |
| `m3_veredictos` | Cinco veredictos por SKU, cobertura, razón, incertidumbre y HLN-DM/BH aparte |

`m3_cartera` publica número de SKU e instancias, valores calculables y no
calculables, media, mediana y percentiles 25 y 75. M4 los lee sin calcular
métricas evaluativas.

La identidad usa la referencia M1, configuración, código, corte y versiones de
política y métricas. No añade otra huella del contenido de la serie. La FK de
ingesta usa la identidad del archivo fuente, distinta de la referencia del
Parquet M1. El protocolo enlaza también la ejecución existente de plataforma;
una corrida iniciada desde el CLI registra esas referencias en la misma base.

## Atomicidad y recuperación

Cada escritura usa `BEGIN IMMEDIATE` y activa claves foráneas. El marcador,
pronósticos, estado y causa de una ventana se confirman juntos. Una unidad
confirmada no se sobrescribe. Los valores no finitos se conservan como NULL,
no como ceros. El estado y sus resultados se escriben en la misma transacción.

```text
EN_CURSO -> EVALUADA -> SELECCIONADA -> PUBLICADA
RECHAZADA: terminal antes de abrir ventanas
```

El protocolo conserva el total previsto. No se pasa a EVALUADA si faltan
unidades. `guardar_evaluacion` comprueba versiones, origen de datos y todas
las ventanas confirmadas, incluida la línea base. Un origen de datos no puede
cambiar para una ingesta ya registrada. PUBLICADA y RECHAZADA son terminales.
La coordinación de recuperación vive en 3.0.

## Publicación y M4

`publicar` exige un veredicto para cada SKU del handoff. Prepara todos los
Parquet y CSV en un directorio temporal y solo entonces coloca el conjunto
completo en su destino. Si falla la preparación, SQLite conserva SELECCIONADA
y no aparece un conjunto parcial nuevo. Una caída después del renombrado y
antes del commit deja una exportación completa no publicada que puede
reconstruirse desde SQLite al reanudar.

M4 consume solo la vista `evidencia_publicada` y tablas de sus corridas
PUBLICADA. La plataforma ofrece `read_m3_evidence` para esa lectura y
`m3_run_status` para progreso y rechazos sin exponer métricas parciales.
CSV es el primer exportable tabular y Parquet conserva los tipos declarados,
incluso en columnas completamente NULL. Las consultas y exportaciones no
vuelven a ejecutar modelos ni a seleccionar campeones.

## Verificación

- `tests/forecasting/persistencia_evidencia/`: estados, integridad,
  idempotencia, migraciones y exportación.
- `tests/forecasting/test_control_evaluacion.py`: flujo real desde M2 y
  recuperación antes y después de confirmar unidades.
- Pruebas DAL de plataforma: visibilidad solo después de publicar y lecturas
  que no modifican la base.

El contrato v1 de presentación conserva su vocabulario histórico. La nueva
lectura M3 expone sus cinco veredictos sin convertirlos en los rótulos antiguos.
La interfaz visual de M4 no forma parte de esta capacidad.
