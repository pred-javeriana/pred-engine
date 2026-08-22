## TASK-DATA-1.2-A1 — Construir sonda de cabeceras LLM y adaptador de mapeo

> **Módulo:** DATA | **Funcionalidad:** 1.2 Alineación Semántica y Validación | **Grupo:** A Alineación Semántica

### **La misión**

- Como Marco de Ingesta de Datos,
- necesito implementar una sonda de cabeceras LLM determinista,
- para que las columnas fuente caóticas se mapeen a nuestro esquema estándar sin reglas hardcodeadas.

### **Pasos**

- Extraer una muestra representativa de las primeras 5 filas del archivo crudo.
- Formular un prompt zero-shot inyectando la muestra e instruyendo a un LLM (con temperatura 0.0 para garantizar determinismo) a mapear las cabeceras al contrato predefinido (`sku_id`, `timestamp`, `demand_qty`, `lead_time_days`).
- Aplicar el diccionario de mapeo generado al DataFrame utilizando operaciones vectorizadas de Pandas (`df.rename()`).
- Descartar forzosamente (Drop) cualquier columna que no haya sido mapeada por el LLM para sanitizar completamente el artefacto de ruido transaccional y optimizar la memoria RAM.

### **Criterios de aceptación**

- [ ] La integración con el LLM es completamente sin estado y maneja correctamente tiempos de espera (timeouts) de la API.
- [ ] Las columnas no mapeadas se eliminan estrictamente, aislando los datos de variables no autorizadas.
- [ ] La funcionalidad funciona según lo definido.
- [ ] El código es modular y demuestra un 90 % de reutilización para verticales alternativas.
- [ ] `README.md`  refleja el estado actual.
- [ ] El ADR está sincronizado entre GitHub y Notion (si aplica).
- [ ] La cobertura de pruebas se verifica en >= 80 %.
- [ ] Los errores se registran mediante el módulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro fundador y se integra mediante *squash merge*.
- [ ] La tarea se mueve a `Completed`.

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
