## TASK-DATA-1.1-A1 — Configurar módulo de registro JSON estructurado

> **Módulo:** DATA | **Funcionalidad:** 1.1 Arquitectura de Extracción y Almacenamiento Crudo | **Grupo:** A Telemetría y Trazabilidad

### **La misión**

- Como Marco de Ingesta de Datos,
- necesito implementar un sistema de registro estructurado en JSON,
- para que cada ejecución del pipeline de extracción deje una trazabilidad auditable y legible por máquinas.

### **Pasos**

- Definir una configuración base del registrador utilizando el módulo estándar `logging` de Python.
- Crear un formateador JSON personalizado para estructurar todas las salidas de registro como cargas útiles JSON en lugar de texto plano.
- Configurar el registrador para incluir estrictamente los campos obligatorios de telemetría (marca de tiempo, módulo de ejecución, hash del archivo y cantidad de filas).
- Enviar los registros JSON a la salida estándar (stdout) para garantizar la compatibilidad con herramientas modernas de observabilidad en la nube.

### **Criterios de aceptación**

- [ ] Los registros de salida son objetos JSON estrictamente válidos.
- [ ] La carga útil del registro incorpora automáticamente la marca de tiempo ISO-8601 y el nivel del registro.
- [ ] La funcionalidad funciona según lo definido.
- [ ] El código es modular y demuestra un 90 % de reutilización para verticales alternativas.
- [ ] `README.md`  reflejan el estado actual.
- [ ] El ADR está sincronizado entre GitHub y Notion (si aplica).
- [ ] La cobertura de pruebas se verifica en >= 80 %.
- [ ] Los errores se registran mediante el módulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro fundador y se integra mediante *squash merge*.
- [ ] La tarea se mueve a `Completed`.

## TASK-DATA-1.1-B1 — Implementar jerarquía de directorios inmutable y permisos

> **Módulo:** DATA | **Funcionalidad:** 1.1 Arquitectura de Extracción y Almacenamiento Crudo | **Grupo:** B Almacenamiento Inmutable y E/S

### **La misión**

- Como Marco de Ingesta de Datos,
- necesito crear las estructuras internas de almacenamiento y aplicar una política de solo lectura,
- para que los archivos fuente sin procesar permanezcan criptográficamente inalterados durante el proceso de ingesta.

### **Pasos**

- Escribir una función utilitaria de configuración que cree dinámicamente el árbol de directorios `/data/raw`, `/data/staging` y `/data/processed` si no existe.
- Implementar un decorador o administrador de contexto para validar las operaciones de E/S e interceptar las operaciones de archivos.
- Bloquear cualquier operación de archivo que intente abrir un archivo dentro de la ruta `/data/raw/` utilizando los modos de escritura (`w`), adición (`a`) o creación exclusiva (`x`).
- Lanzar un `PermissionError` personalizado cuando ocurra una violación de escritura, registrando la ruta exacta y la función que realizó la llamada.

### **Criterios de aceptación**

- [ ] El script de generación de directorios se ejecuta sin estado y no falla si los directorios ya existen.
- [ ] Cualquier intento programático de escribir en `/data/raw/` genera una excepción explícita y detiene la ejecución.
- [ ] La funcionalidad funciona según lo definido.
- [ ] El código es modular y demuestra un 90 % de reutilización para verticales alternativas.
- [ ] `README.md` y `API_SPECIFICATION.md` reflejan el estado actual.
- [ ] El ADR está sincronizado entre GitHub y Notion (si aplica).
- [ ] La cobertura de pruebas se verifica en >= 80 %.
- [ ] Los errores se registran mediante el módulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro fundador y se integra mediante *squash merge*.
- [ ] La tarea se mueve a `Completed`.

## TASK-DATA-1.1-B2 — Construir extractor CSV base y exportador Parquet

> **Módulo:** DATA | **Funcionalidad:** 1.1 Arquitectura de Extracción y Almacenamiento Crudo | **Grupo:** B Almacenamiento Inmutable y E/S

### **La misión**

- Como Marco de Ingesta de Datos,
- necesito construir un adaptador de E/S sin estado utilizando Pandas y PyArrow,
- para que los archivos CSV sin procesar puedan leerse de forma segura y exportarse como artefactos Parquet columnares altamente comprimidos.

### **Pasos**

- Implementar una función de extracción pasiva que lea un archivo `.csv` desde `/data/raw/` en un DataFrame de Pandas sin inferir fechas ni modificar los tipos de datos.
- Calcular y devolver el hash SHA-256 del archivo sin procesar ingerido junto con el DataFrame para garantizar la auditabilidad.
- Implementar una función de exportación independiente que reciba un DataFrame de Pandas y lo guarde en `/data/processed/` utilizando el formato `.parquet` mediante el motor `pyarrow`.
- Garantizar que todas las funciones sean puras (sin estado) y no conserven datos en memoria global después de devolver el artefacto.

### **Criterios de aceptación**

- [ ] La función de extracción CSV lee de forma segura conjuntos de datos de más de 50.000 filas sin provocar un desbordamiento de memoria.
- [ ] La función de exportación escribe correctamente archivos `.parquet` comprimidos conservando intactos los nombres de las columnas.
- [ ] La funcionalidad funciona según lo definido.
- [ ] El código es modular y demuestra un 90 % de reutilización para verticales alternativas.
- [ ] `README.md` y `API_SPECIFICATION.md` reflejan el estado actual.
- [ ] El ADR está sincronizado entre GitHub y Notion (si aplica).
- [ ] La cobertura de pruebas se verifica en >= 80 %.
- [ ] Los errores se registran mediante el módulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro fundador y se integra mediante *squash merge*.
- [ ] La tarea se mueve a `Completed`.
