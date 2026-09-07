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
