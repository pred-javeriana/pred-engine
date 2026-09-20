## TASK-DATA-1.2-A1 — Construir probe de cabecera LLM diagnostico y generador de feedback

> **Modulo:** DATA | **Funcionalidad:** 1.2 Alineacion Semantica y Validacion | **Grupo:** A Alineacion Semantica

### La mision

- Como Marco de Ingesta de Datos,
- necesito implementar un asistente LLM diagnostico,
- para evaluar archivos fuente caoticos frente a nuestro esquema estricto y
  proporcionar a los usuarios feedback JSON accionable para corregir sus datos.

### Pasos

- Extraer una muestra representativa de las primeras 5 filas del DataFrame original.
- Formular un prompt zero-shot que instruya a un LLM, con `temperature=0.0`, a
  comparar las cabeceras de origen contra el contrato predefinido.
- Configurar el LLM para producir un payload JSON estricto con `status` y
  `diagnostic[]`.
- Implementar fail-fast: si `rejected`, registrar/imprimir JSON y detener el
  pipeline. **NO modificar el DataFrame.**

### Criterios de aceptacion

- [x] La integracion con el LLM es stateless y maneja timeouts.
- [x] Cero operaciones de mutacion (`df.rename`, `df.drop`).
- [x] Salida JSON estructurada con instrucciones de correccion.
- [x] Cobertura >= 80 % via `uv run pytest`.
- [x] Errores registrados via `logging` JSON.

## TASK-DATA-1.2-B1 — Implementar validacion estricta de esquemas con Pydantic

> **Modulo:** DATA | **Funcionalidad:** 1.2 | **Grupo:** B Validacion de Esquemas

Implementado en `pred_engine.ingesta.validador_formato` (`InventoryObservation`,
`validate_aligned_frame`). Requiere cabeceras canonicas exactas post-sonda.

## TASK-DATA-1.2-C1 — Desarrollar motor de remuestreo temporal sin estado

> **Modulo:** DATA | **Funcionalidad:** 1.2 | **Grupo:** C Continuidad Temporal

Implementado en `pred_engine.ingesta.continuidad` (`resample_daily`).

## TASK-DATA-1.2-A1 — Construir probe de cabecera LLM diagnóstico y generador de feedback

> **Módulo:** DATA | **Feature:** 1.2 Alineación Semántica y Validación | **Grupo:** A Semantic Alignment

### La misión

- Como **Data Ingestion Framework**,
- necesito implementar un asistente LLM diagnóstico,
- para evaluar archivos fuente caóticos frente a nuestro esquema estricto y proporcionar a los usuarios feedback JSON accionable para corregir sus datos.

### Pasos

- Extraer una muestra representativa de las primeras 5 filas del `DataFrame` original.
- Formular un prompt zero-shot que instruya a un LLM, con `temperature=0.0`, a comparar las cabeceras de origen contra el contrato predefinido:
    - `sku_id`
    - `timestamp`
    - `demand_qty`
    - `lead_time_days`
- Configurar el LLM para producir un payload JSON estricto que indique:
    - `"status": "accepted"` o `"status": "rejected"`.
    - Un array `"diagnostic"` que detalle exactamente qué columnas faltan.
    - Instrucciones explícitas sobre cómo renombrar las columnas existentes para cumplir el contrato.
- Implementar un mecanismo **fail-fast**:
    - Si el estado es `"rejected"`, imprimir o registrar el reporte JSON de diagnóstico para el usuario.
    - Detener inmediatamente el pipeline.
    - **NO modificar el `DataFrame`.**

### Criterios de aceptación

- [ ] La integración con el LLM funciona exclusivamente como herramienta de diagnóstico stateless y ejecuta cero operaciones de mutación (`df.rename`, `df.drop`) sobre el dataset.
- [ ] La salida es una respuesta JSON estructurada que indica claramente al usuario cómo corregir su archivo CSV.
- [ ] La funcionalidad funciona según lo definido.
- [ ] El código es modular y demuestra un **90 % de reutilización** para verticales alternativas.
- [ ] `README.md` y `API_SPECIFICATION.md` reflejan el estado actual de la implementación.
- [ ] El ADR está sincronizado entre GitHub y Notion, si aplica.
- [ ] La cobertura de tests está verificada en **>= 80 %** mediante el comando `make test-cov`. **No utilizar los valores predeterminados de `pytest` directamente.**
- [ ] El logging estructurado y el tracking de errores mediante Sentry están activos.
- [ ] El PR ha sido aprobado por al menos otro fundador y fusionado mediante **squash merge**.
- [ ] La tarea se mueve a **"Completed"**.

## TASK-DATA-1.2-B1 — Implementar validación estricta de esquemas con Pydantic

> **Módulo:** DATA | **Funcionalidad:** 1.2 Alineación Semántica y Validación | **Grupo:** B Validación de Esquemas

### **La misión**

- Como Marco de Ingesta de Datos,
- necesito imponer la integridad matemática y estructural usando Pydantic,
- para que los estados inválidos se bloqueen antes de llegar a los modelos de pronóstico.

### **Pasos**

- Definir un contrato estricto de Pydantic v2 especificando que `timestamp` debe ser un objeto `datetime64` válido.
- Definir en el contrato que la demanda (`demand_qty`) debe ser estrictamente numérica y `>= 0`.
- Definir en el contrato que el tiempo de entrega (`lead_time_days`) debe ser estrictamente numérico y `>= 1`.
- Implementar una barrera de validación *fail-fast* que intercepte el DataFrame inmediatamente después de la alineación semántica.
- Levantar una excepción crítica, detener la ejecución y registrar la violación exacta del esquema en los logs estructurados si se incumple alguna regla.

### **Criterios de aceptación**

- [ ] El framework de validación (Pydantic) impone estrictamente los tipos y límites matemáticos sin realizar coerción implícita insegura.
- [ ] La ejecución se detiene inmediatamente (_fail-fast_) al detectar demandas negativas o fechas inválidas.
- [ ] La funcionalidad funciona según lo definido.
- [ ] El código es modular y demuestra un 90 % de reutilización para verticales alternativas.
- [ ] `README.md`  refleja el estado actual.
- [ ] El ADR está sincronizado entre GitHub y Notion (si aplica).
- [ ] La cobertura de pruebas se verifica en >= 80 %.
- [ ] Los errores se registran mediante el módulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro fundador y se integra mediante *squash merge*.
- [ ] La tarea se mueve a `Completed`.

## TASK-DATA-1.2-C1 — Desarrollar motor de remuestreo temporal sin estado

> **Módulo:** DATA | **Funcionalidad:** 1.2 Alineación Semántica y Validación | **Grupo:** C Continuidad Temporal

### **La misión**

- Como Marco de Ingesta de Datos,
- necesito imponer la continuidad temporal en datos de inventario esporádicos,
- para que los modelos de pronóstico reciban cuadrículas de tiempo matemáticamente válidas y equidistantes.

### **Pasos**

- Escribir una función sin estado que reciba el DataFrame validado por Pydantic y agrupe los datos por `sku_id`.
- Generar una cuadrícula temporal continua (resolución diaria) para cada identificador único, abarcando desde su fecha mínima hasta su máxima histórica.
- Ejecutar un *Left Join* de los datos transaccionales validados sobre esta cuadrícula temporal continua.
- Imputar explícitamente todos los valores nulos (NaN) resultantes en la columna de demanda con un 0 (Cero matemático) para preservar la esparsidad de la serie (Zero-Filling).

### **Criterios de aceptación**

- [ ] La cuadrícula temporal generada contiene estrictamente intervalos diarios equidistantes sin omitir ninguna fecha.
- [ ] Los períodos sin demanda se imputan exitosamente con `0`, evitando caídas matemáticas en los modelos posteriores.
- [ ] La funcionalidad funciona según lo definido.
- [ ] El código es modular y demuestra un 90 % de reutilización para verticales alternativas.
- [ ] `README.md` reflejan el estado actual.
- [ ] El ADR está sincronizado entre GitHub y Notion (si aplica).
- [ ] La cobertura de pruebas se verifica en >= 80 %.
- [ ] Los errores se registran mediante el módulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro fundador y se integra mediante *squash merge*.
- [ ] La tarea se mueve a `Completed`.
