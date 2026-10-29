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
- `esquema_m3_0003.sql`: procedencia M1 por CSV y huella del Parquet.
- `esquema_m3_0004.sql`: distingue ausencia de declaración de datos reales.
- `esquema_m3_0005.sql`: la vista `evidencia_publicada` expone
  `origen_declarado` junto a `datos_sinteticos`.
- `esquema_m3_0006.sql`: los datos que M4 muestra y M3 publica con la
  corrida: corte, lanzamientos tardíos, demanda observada y la vista
  `m3_campeones`.
- `esquema_m3_0007.sql`: la evidencia del bloque de prueba (ADR-023):
  significancia por categoría, confirmación del campeón, comparaciones entre
  familias, selección de robustez, intervalos de la razón y exclusiones de M2.

La plataforma incorpora copias exactas como migraciones 0005 a 0011.
La plataforma aplica `esquema_m3.sql` como su migración 0005 (el comentario histórico del archivo dice 0002).
Las versiones anteriores se conservan sin cambios. La versión vigente es 11.
`aplicar_migraciones_m3` usa el mismo `PRAGMA user_version`. El CLI aplica las
pendientes sobre una base existente con las migraciones previas de plataforma.
No modifica las versiones 0001 a 0004 y rechaza una versión desconocida.
El comentario histórico del esquema inicial menciona 0002; la numeración
vigente y su motivo se documentan en ADR-021.

## Evidencia durable

| Tabla | Contenido |
| --- | --- |
| `m3_ingestas` | Procedencia real/sintética inmutable por identidad CSV y Parquet M1 |
| `m3_corridas` | Identidad, referencia de ingesta fuente, corrida M2, origen de datos, estado y rechazo |
| `m3_protocolos` | Protocolo y handoff congelados, apertura, procedencia, ejecución de plataforma y unidades previstas |
| `m3_skus` | SKU del handoff, incluidos los que solo tienen fallos de adaptación |
| `m3_fallos` | Fallos del adaptador con candidato, campo y causa |
| `m3_unidades` | Marcador por candidato y origen, estado, causa, emisión, código y semilla |
| `m3_pronosticos` | Fechas y valores por origen y paso; NULL cuando el pronóstico falló |
| `m3_metricas` | MAE, RMSE, MASE, ME, razón y cobertura: por ventana en toda la reserva y agregadas del bloque de prueba |
| `m3_categorias` | Selección en el bloque de selección, exclusiones, medianas y conteos de veredictos |
| `m3_cartera` | Distribuciones precalculadas del bloque de prueba por clase, familia y métrica |
| `m3_veredictos` | Cinco veredictos por SKU en el bloque de prueba, cobertura, razón, incertidumbre y HLN-DM aparte, con BH dentro de la categoría |
| `m3_cortes` | Corte t* con que L2 separó la reserva, leído de los metadatos de M1 |
| `m3_lanzamientos_tardios` | Productos que M1 excluyó por lanzarse después del corte, con su causa |
| `m3_exclusiones` | Productos que M2 excluyó antes de L2 o al cerrar L2, con categoría, código `eligibility.*` y causa |
| `m3_observaciones` | Demanda real de cada día de la reserva por SKU; NULL sin observación |
| `m3_significancia` | Por categoría: SKU probados, mejores y peores que Seasonal Naive con BH y con BY, y nivel FCR |
| `m3_confirmaciones` | Por categoría: estado de la confirmación del campeón, p de intersección-unión y de Holm, mediana de r frente a Seasonal Naive con su intervalo y la causa de un valor sin definir, y contendiente |
| `m3_comparaciones` | Campeón frente a cada competidora: SKU, mediana de razones con intervalo bootstrap, Hodges-Lehmann, victorias con intervalo de Wilson, Wilcoxon, conteos HLN-DM y causa |
| `m3_seleccion_previa` | La regla de selección sobre los pliegues L3, como control de robustez |
| `m3_intervalos_r` | Intervalo de r de cada SKU probado y, si BH lo selecciona, al nivel FCR; `inferior_cola` marca un conjunto de dos tramos |
| `m3_campeones` (vista) | SKU por familia campeona, evaluados y no evaluados, por clase y cartera |

Los límites de los bloques están en `m3_protocolos.protocolo`, clave
`bloques`: fracción, días, horizonte y el primer y último origen y día de cada
bloque.

`m3_cartera` publica número de SKU e instancias, valores calculables y no
calculables, media, mediana y percentiles 25 y 75. M4 los lee sin calcular
métricas evaluativas.

El corte, los lanzamientos tardíos y las exclusiones de M2 se fijan en la
transacción del protocolo, con los mismos t* y exclusiones que usó L2, sin
recalcularlos. Una reanudación usa las exclusiones de M2 de la procedencia
durable. Un SKU lanzado después del corte o excluido por M2 nunca entra en
`m3_skus`; dos disparadores por tabla lo impiden en ambos sentidos. La demanda observada se escribe con la evaluación,
desde la misma reserva que puntuó las métricas, para el calendario completo de
la reserva y para cada SKU del manifiesto. Cada fecha pronosticada tiene su
fila. `m3_campeones` es una vista sobre `m3_veredictos`: funciona también para
corridas publicadas antes de la versión 10 y no tiene escritura propia. Las
filas con `sku_class` NULL son la cartera completa.

Una corrida que pasó de EVALUADA antes de la versión 10 no recibe corte ni
demanda observada al reanudarse: M4 responde `metadata_missing` para ella.

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
completo en su destino. Lee cada tabla en orden de `rowid`, en bloques de
10.000 filas, y mantiene abiertos un escritor Parquet y un flujo CSV por tabla.
Cada bloque conserva los tipos declarados en SQLite, incluso si está vacío o
solo contiene NULL. No carga todas las tablas de la corrida en memoria.
Si falla la preparación, SQLite conserva SELECCIONADA
y no aparece un conjunto parcial nuevo. Una caída después del renombrado y
antes del commit deja una exportación completa no publicada que puede
reconstruirse desde SQLite al reanudar.
Antes de reexportar una corrida SELECCIONADA, el motor elimina los directorios
temporales de intentos anteriores de esa corrida. Lo hace con el bloqueo de
escritura tomado para no borrar una exportación en curso. Los temporales de
otras corridas se conservan.

M4 consume solo la vista `evidencia_publicada` y tablas de sus corridas
PUBLICADA. En pred-platform, `DataRepository` ofrece esa lectura como
documentos del contrato de lectura M3 v2 (`get_m3_evidence` y las demás
lecturas por tabla) y `get_m3_run_status` para progreso y rechazos sin exponer
métricas parciales.
CSV es el primer exportable tabular y Parquet conserva los tipos declarados,
incluso en columnas completamente NULL. Las consultas y exportaciones no
vuelven a ejecutar modelos ni a seleccionar campeones.

## Verificación

- `tests/forecasting/persistencia_evidencia/`: estados, integridad,
  idempotencia, migraciones y exportación.
- `tests/forecasting/test_control_evaluacion.py`: flujo real desde M2 y
  recuperación antes y después de confirmar unidades, también del corte y de
  la demanda observada.
- `tests/forecasting/test_publicacion_m4.py`: `pred-engine run` publica corte,
  lanzamientos tardíos, demanda observada y campeones.
- Pruebas DAL de plataforma: visibilidad solo después de publicar y lecturas
  que no modifican la base.

El contrato v1 de presentación usa sus propios rótulos. La lectura M3 expone
los cinco veredictos sin convertirlos a esos rótulos.
La interfaz visual de M4 no forma parte de esta capacidad.
