## TASK-DATA-1.3-A1 — Documentar la especificacion de la API para el motor de topologia

> **Modulo:** DATA | **Funcionalidad:** 1.3 Motor de Topologia de SKU | **Grupo:** A Diseno del contrato

### Criterios de aceptacion

- [x] El documento incluye los umbrales exactos de ADI y CV² definidos en la literatura academica.
- [x] La salida especificada garantiza la inmutabilidad de las columnas transaccionales originales.
- [x] La funcionalidad funciona segun lo definido.
- [x] El codigo es modular y demuestra un 90 % de reutilizacion para verticales alternativas.
- [x] `README.md` y `API_SPECIFICATION.md` reflejan el estado actual.
- [ ] El ADR esta sincronizado entre GitHub y Notion (si aplica).
- [x] La cobertura de pruebas se verifica en >= 80 %.
- [x] Los errores se registran mediante el modulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro fundador y se integra mediante _squash merge_.
- [ ] La tarea se mueve a `Completed`.
- [x] No se registra nueva informacion de identificacion personal (PII/PHI) en los registros.
- [x] La ejecucion pasa `ruff check .` y `ruff format .` con cero errores.
- [x] El tipado estricto es validado por `pyright` sin errores.
- [ ] Los _pre-commit hooks_ estan instalados y verificados en el entorno local.

## TASK-DATA-1.3-B1 — Implementar la funcion sin estado para el calculo de ADI

> **Modulo:** DATA | **Funcionalidad:** 1.3 | **Grupo:** B Nucleo matematico

Implementado en `pred_engine.ingesta.categorizacion.adi` (`compute_adi`).

### Criterios de aceptacion

- [x] La funcion ignora los valores nulos o negativos al contar la demanda activa.
- [x] El calculo gestiona de forma segura las divisiones por cero devolviendo un error tipado.
- [x] La funcionalidad funciona segun lo definido.
- [x] El codigo es modular y demuestra un 90 % de reutilizacion para verticales alternativas.
- [x] `README.md` y `API_SPECIFICATION.md` reflejan el estado actual.
- [ ] El ADR esta sincronizado entre GitHub y Notion (si aplica).
- [x] La cobertura de pruebas se verifica en >= 80 %.
- [x] Los errores se registran mediante el modulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro fundador y se integra mediante _squash merge_.
- [ ] La tarea se mueve a `Completed`.
- [x] No se registra nueva informacion de identificacion personal (PII/PHI) en los registros.
- [x] La ejecucion pasa `ruff check .` y `ruff format .` con cero errores.
- [x] El tipado estricto es validado por `pyright` sin errores.
- [ ] Los _pre-commit hooks_ estan instalados y verificados en el entorno local.

## TASK-DATA-1.3-B2 — Implementar la funcion sin estado para el calculo de CV²

> **Modulo:** DATA | **Funcionalidad:** 1.3 | **Grupo:** B Nucleo matematico

Implementado en `pred_engine.ingesta.categorizacion.cv2` (`compute_cv2`).

### Criterios de aceptacion

- [x] El calculo excluye estrictamente los periodos con demanda igual a cero.
- [x] El valor de retorno es un numero de punto flotante con tipado estático.
- [x] La funcionalidad funciona segun lo definido.
- [x] El codigo es modular y demuestra un 90 % de reutilizacion para verticales alternativas.
- [x] `README.md` y `API_SPECIFICATION.md` reflejan el estado actual.
- [ ] El ADR esta sincronizado entre GitHub y Notion (si aplica).
- [x] La cobertura de pruebas se verifica en >= 80 %.
- [x] Los errores se registran mediante el modulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro fundador y se integra mediante _squash merge_.
- [ ] La tarea se mueve a `Completed`.
- [x] No se registra nueva informacion de identificacion personal (PII/PHI) en los registros.
- [x] La ejecucion pasa `ruff check .` y `ruff format .` con cero errores.
- [x] El tipado estricto es validado por `pyright` sin errores.
- [ ] Los _pre-commit hooks_ estan instalados y verificados en el entorno local.

## TASK-DATA-1.3-C1 — Implementar el enrutador de la matriz de decision de Syntetos-Boylan

> **Modulo:** DATA | **Funcionalidad:** 1.3 | **Grupo:** C Logica de enrutamiento

Implementado en `pred_engine.ingesta.categorizacion.enrutador` (`route_syntetos_boylan`).

### Criterios de aceptacion

- [x] Los umbrales de evaluacion respetan los limites estrictos: 1.32 para ADI y 0.49 para CV².
- [x] Los empates en los umbrales se resuelven segun la especificacion academica (cuadrante alto).
- [x] La funcionalidad funciona segun lo definido.
- [x] El codigo es modular y demuestra un 90 % de reutilizacion para verticales alternativas.
- [x] `README.md` y `API_SPECIFICATION.md` reflejan el estado actual.
- [ ] El ADR esta sincronizado entre GitHub y Notion (si aplica).
- [x] La cobertura de pruebas se verifica en >= 80 %.
- [x] Los errores se registran mediante el modulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro fundador y se integra mediante _squash merge_.
- [ ] La tarea se mueve a `Completed`.
- [x] No se registra nueva informacion de identificacion personal (PII/PHI) en los registros.
- [x] La ejecucion pasa `ruff check .` y `ruff format .` con cero errores.
- [x] El tipado estricto es validado por `pyright` sin errores.
- [ ] Los _pre-commit hooks_ estan instalados y verificados en el entorno local.

## TASK-DATA-1.3-C2 — Inyectar las categorias de topologia en el flujo de datos

> **Modulo:** DATA | **Funcionalidad:** 1.3 | **Grupo:** C Logica de enrutamiento

Implementado en `pred_engine.ingesta.categorizacion.panel` (`classify_panel`) e
integrado en `pred_engine.ingesta.pipeline`.

### Criterios de aceptacion

- [x] La columna inyectada se denomina estrictamente `sku_class`.
- [x] Un SKU conserva exactamente la misma etiqueta en todas sus filas, sin realizar recalculos por fila.
- [x] La funcionalidad funciona segun lo definido.
- [x] El codigo es modular y demuestra un 90 % de reutilizacion para verticales alternativas.
- [x] `README.md` y `API_SPECIFICATION.md` reflejan el estado actual.
- [ ] El ADR esta sincronizado entre GitHub y Notion (si aplica).
- [x] La cobertura de pruebas se verifica en >= 80 %.
- [x] Los errores se registran mediante el modulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro fundador y se integra mediante _squash merge_.
- [ ] La tarea se mueve a `Completed`.
- [x] No se registra nueva informacion de identificacion personal (PII/PHI) en los registros.
- [x] La ejecucion pasa `ruff check .` y `ruff format .` con cero errores.
- [x] El tipado estricto es validado por `pyright` sin errores.
- [ ] Los _pre-commit hooks_ estan instalados y verificados en el entorno local.

## TASK-DATA-1.3-A1 — Documentar la especificación de la API para el motor de topología

> **Módulo:** DATA | **Funcionalidad:** 1.3 Motor de Topología de SKU | **Grupo:** A Diseño del contrato

### La misión

- Como Arquitecto de Datos,
- necesito definir el contrato de entrada y salida del motor de topología,
- para que los equipos de modelado dependan de esquemas inmutables y predecibles.

### Pasos

- Crear el documento de especificación de la API dentro del directorio de la funcionalidad.
- Definir las columnas de entrada obligatorias y las restricciones matemáticas para el procesamiento.
- Especificar el esquema de salida, incluida la bandera categórica de topología.
- Revisar los límites del contrato con el responsable de infraestructura para evitar cambios incompatibles.

### Criterios de aceptación

- [ ] El documento incluye los umbrales exactos de ADI y CV² definidos en la literatura académica.
- [ ] La salida especificada garantiza la inmutabilidad de las columnas transaccionales originales.
- [ ] La funcionalidad funciona según lo definido.
- [ ] El código es modular y demuestra un 90 % de reutilización para verticales alternativas.
- [ ] `README.md` y `API_SPECIFICATION.md` reflejan el estado actual.
- [ ] El ADR está sincronizado entre GitHub y Notion (si aplica).
- [ ] La cobertura de pruebas se verifica en \>= 80 %.
- [ ] Los errores se registran mediante el módulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro fundador y se integra mediante _squash merge_.
- [ ] La tarea se mueve a `Completed`.
- [ ] No se registra nueva información de identificación personal (PII/PHI) en los registros.
- [ ] La ejecución pasa `ruff check .` y `ruff format .` con cero errores.
- [ ] El tipado estricto es validado por `pyright` sin errores.
- [ ] Los _pre-commit hooks_ están instalados y verificados en el entorno local.

---

## TASK-DATA-1.3-B1 — Implementar la función sin estado para el cálculo de ADI

> **Módulo:** DATA | **Funcionalidad:** 1.3 Motor de Topología de SKU | **Grupo:** B Núcleo matemático

### La misión

- Como Ingeniero de Datos,
- necesito desarrollar una función pura para calcular el ADI,
- para que el sistema evalúe la dispersión temporal sin modificar el estado de los datos.

### Pasos

- Definir una función sin estado que acepte un arreglo numérico de demandas históricas.
- Calcular el número total de períodos temporales evaluados.
- Contar las ocurrencias de demanda estrictamente positivas.
- Devolver la razón de Intervalo Promedio de Demanda (ADI) calculada.

### Criterios de aceptación

- [ ] La función ignora los valores nulos o negativos al contar la demanda activa.
- [ ] El cálculo gestiona de forma segura las divisiones por cero devolviendo un error tipado.
- [ ] La funcionalidad funciona según lo definido.
- [ ] El código es modular y demuestra un 90 % de reutilización para verticales alternativas.
- [ ] `README.md` y `API_SPECIFICATION.md` reflejan el estado actual.
- [ ] El ADR está sincronizado entre GitHub y Notion (si aplica).
- [ ] La cobertura de pruebas se verifica en \>= 80 %.
- [ ] Los errores se registran mediante el módulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro fundador y se integra mediante _squash merge_.
- [ ] La tarea se mueve a `Completed`.
- [ ] No se registra nueva información de identificación personal (PII/PHI) en los registros.
- [ ] La ejecución pasa `ruff check .` y `ruff format .` con cero errores.
- [ ] El tipado estricto es validado por `pyright` sin errores.
- [ ] Los _pre-commit hooks_ están instalados y verificados en el entorno local.

---

## TASK-DATA-1.3-B2 — Implementar la función sin estado para el cálculo de CV²

> **Módulo:** DATA | **Funcionalidad:** 1.3 Motor de Topología de SKU | **Grupo:** B Núcleo matemático

### La misión

- Como Ingeniero de Datos,
- necesito desarrollar una función pura para calcular el CV²,
- para que el sistema mida la volatilidad aplicando los filtros matemáticos correctos.

### Pasos

- Definir una función sin estado que acepte un arreglo numérico de demandas.
- Filtrar el conjunto de datos para aislar los períodos con demanda estrictamente positiva.
- Calcular la media aritmética y la desviación estándar del subconjunto filtrado.
- Devolver el cuadrado de la razón entre la desviación estándar y la media.

### Criterios de aceptación

- [ ] El cálculo excluye estrictamente los períodos con demanda igual a cero.
- [ ] El valor de retorno es un número de punto flotante con tipado estático.
- [ ] La funcionalidad funciona según lo definido.
- [ ] El código es modular y demuestra un 90 % de reutilización para verticales alternativas.
- [ ] `README.md` y `API_SPECIFICATION.md` reflejan el estado actual.
- [ ] El ADR está sincronizado entre GitHub y Notion (si aplica).
- [ ] La cobertura de pruebas se verifica en \>= 80 %.
- [ ] Los errores se registran mediante el módulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro fundador y se integra mediante _squash merge_.
- [ ] La tarea se mueve a `Completed`.
- [ ] No se registra nueva información de identificación personal (PII/PHI) en los registros.
- [ ] La ejecución pasa `ruff check .` y `ruff format .` con cero errores.
- [ ] El tipado estricto es validado por `pyright` sin errores.
- [ ] Los _pre-commit hooks_ están instalados y verificados en el entorno local.

---

## TASK-DATA-1.3-C1 — Implementar el enrutador de la matriz de decisión de Syntetos-Boylan

> **Módulo:** DATA | **Funcionalidad:** 1.3 Motor de Topología de SKU | **Grupo:** C Lógica de enrutamiento

### La misión

- Como Ingeniero de Datos,
- necesito programar el enrutador categórico basado en umbrales,
- para que cada producto sea asignado al cuadrante topológico correcto.

### Pasos

- Definir una función de enrutamiento que acepte los valores matemáticos de ADI y CV².
- Aplicar la condición matemática para la categoría Suave.
- Aplicar la condición matemática para la categoría Intermitente.
- Aplicar la condición matemática para la categoría Errática.
- Aplicar la condición matemática para la categoría Irregular.
- Devolver la cadena correspondiente a la categoría topológica.

### Criterios de aceptación

- [ ] Los umbrales de evaluación respetan los límites estrictos: 1.32 para ADI y 0.49 para CV².
- [ ] Los empates en los umbrales se resuelven según la especificación académica (por ejemplo, ADI = 1.32 cae en Intermitente o Irregular).
- [ ] La funcionalidad funciona según lo definido.
- [ ] El código es modular y demuestra un 90 % de reutilización para verticales alternativas.
- [ ] `README.md` y `API_SPECIFICATION.md` reflejan el estado actual.
- [ ] El ADR está sincronizado entre GitHub y Notion (si aplica).
- [ ] La cobertura de pruebas se verifica en \>= 80 %.
- [ ] Los errores se registran mediante el módulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro fundador y se integra mediante _squash merge_.
- [ ] La tarea se mueve a `Completed`.
- [ ] No se registra nueva información de identificación personal (PII/PHI) en los registros.
- [ ] La ejecución pasa `ruff check .` y `ruff format .` con cero errores.
- [ ] El tipado estricto es validado por `pyright` sin errores.
- [ ] Los _pre-commit hooks_ están instalados y verificados en el entorno local.

---

## TASK-DATA-1.3-C2 — Inyectar las categorías de topología en el flujo de datos

> **Módulo:** DATA | **Funcionalidad:** 1.3 Motor de Topología de SKU | **Grupo:** C Lógica de enrutamiento

### La misión

- Como Ingeniero de Datos,
- necesito integrar las funciones de topología en el flujo principal,
- para que el conjunto de datos final cuente con el enriquecimiento analítico requerido.

### Pasos

- Cargar la cuadrícula de datos semánticamente alineada y remuestreada.
- Agrupar los registros temporales por identificador único.
- Aplicar las funciones del núcleo matemático a cada subconjunto agrupado.
- Inyectar la categoría generada en una nueva columna estructurada.

### Criterios de aceptación

- [ ] La columna inyectada se denomina estrictamente `sku_class`.
- [ ] Un SKU conserva exactamente la misma etiqueta en todas sus filas, sin realizar recálculos por fila.
- [ ] La funcionalidad funciona según lo definido.
- [ ] El código es modular y demuestra un 90 % de reutilización para verticales alternativas.
- [ ] `README.md` y `API_SPECIFICATION.md` reflejan el estado actual.
- [ ] El ADR está sincronizado entre GitHub y Notion (si aplica).
- [ ] La cobertura de pruebas se verifica en \>= 80 %.
- [ ] Los errores se registran mediante el módulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro fundador y se integra mediante _squash merge_.
- [ ] La tarea se mueve a `Completed`.
- [ ] No se registra nueva información de identificación personal (PII/PHI) en los registros.
- [ ] La ejecución pasa `ruff check .` y `ruff format .` con cero errores.
- [ ] El tipado estricto es validado por `pyright` sin errores.
- [ ] Los _pre-commit hooks_ están instalados y verificados en el entorno local.
