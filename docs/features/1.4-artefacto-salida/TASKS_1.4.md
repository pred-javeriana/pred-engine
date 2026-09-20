## TASK-DATA-1.4-A1 — Implementar validador del contrato final

Implementado en `pred_engine.ingesta.salida.validador` y
`ClassifiedObservation`.

### Criterios de aceptacion

- [x] El validador rechaza cualquier DataFrame que no contenga exactamente el contrato de cinco columnas.
- [x] `sku_id` cumple el tipo String definido por el contrato.
- [x] `timestamp` cumple el tipo Datetime64 y no contiene valores nulos.
- [x] `demand_qty` cumple el tipo Float, no contiene valores nulos y todos sus valores son >= 0.
- [x] `lead_time_days` cumple el tipo Integer y respeta las restricciones heredadas de 1.2.
- [x] `sku_class` solo admite `smooth`, `intermittent`, `erratic` y `lumpy`.
- [x] Cada SKU posee exactamente una clasificacion constante en todas sus filas.
- [x] La validacion opera de forma fail-closed y no modifica el DataFrame recibido.
- [x] La funcionalidad funciona segun lo definido.
- [x] El codigo es modular y demuestra un 90 % de reutilizacion para verticales alternativas.
- [x] `README.md` y `API_SPECIFICATION.md` reflejan el estado actual.
- [x] Los ADR-01-009, ADR-01-010, ADR-01-011 y ADR-01-012 reflejan el estado final de la implementacion.
- [x] La cobertura de pruebas se verifica en >= 80 % mediante `make test-cov`.
- [x] Los errores se registran mediante el modulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro integrante y se integra mediante _squash merge_.
- [ ] La tarea se mueve a `Completed`.
- [x] No se registra nueva informacion de identificacion personal (PII/PHI) en los registros.
- [x] La ejecucion pasa `ruff check` sobre el codigo de 1.4 y `ruff format`.
- [x] El tipado estricto es validado por `pyright` sin errores.
- [ ] Los _pre-commit hooks_ estan instalados y verificados en el entorno local.

## TASK-DATA-1.4-A2 — Verificar preservacion del panel clasificado

Implementado en `pred_engine.ingesta.salida.preservacion`. Se ejecuta
dentro de `publish_classified_panel` antes de escribir.

## TASK-DATA-1.4-B1 — Publicar artefacto final en Parquet

Implementado en `pred_engine.ingesta.salida.publicador`.
Ruta `{data_root}/processed/<stem>.parquet`.

## TASK-DATA-1.4-C1 — Validar handoff completo de ingesta

Implementado en `run_classify_csv` / `run_ingest` + `run_verify_parquet`.
Prueba de integracion: `tests/ingesta/salida/test_handoff.py`.
Dataset oficial: 4862 filas, 10 SKU, todos `intermittent`.

## TASK-DATA-1.4-A1 — Implementar validador del contrato final

> **Módulo:** DATA | **Funcionalidad:** 1.4 Contrato de Datos Final y Artefacto de Salida | **Grupo:** A Contrato de salida

### La misión

- Como Pipeline de Ingestión,
- necesito validar estrictamente el panel clasificado recibido desde 1.3,
- para que únicamente artefactos compatibles con los Módulos 2 y 3 puedan ser publicados.

### Pasos

- Validar la presencia exacta de `sku_id`, `timestamp`, `demand_qty`, `lead_time_days` y `sku_class`.
- Validar los tipos establecidos por el contrato final para cada columna.
- Validar ausencia de valores nulos en las columnas obligatorias.
- Validar que `demand_qty` sea no negativa y que `lead_time_days` respete el dominio establecido.
- Validar que `sku_class` pertenezca exclusivamente al conjunto de etiquetas permitido.
- Validar que `sku_class` sea constante dentro de cada `sku_id`.
- Rechazar el panel completo ante cualquier incumplimiento antes de persistirlo.

### Criterios de aceptación

- [ ] El validador rechaza cualquier DataFrame que no contenga exactamente el contrato de cinco columnas.
- [ ] `sku_id` cumple el tipo String definido por el contrato.
- [ ] `timestamp` cumple el tipo Datetime64 y no contiene valores nulos.
- [ ] `demand_qty` cumple el tipo Float, no contiene valores nulos y todos sus valores son >= 0.
- [ ] `lead_time_days` cumple el tipo Integer y respeta las restricciones heredadas de 1.2.
- [ ] `sku_class` solo admite `smooth`, `intermittent`, `erratic` y `lumpy`.
- [ ] Cada SKU posee exactamente una clasificación constante en todas sus filas.
- [ ] La validación opera de forma fail-closed y no modifica el DataFrame recibido.
- [ ] La funcionalidad funciona según lo definido.
- [ ] El código es modular y demuestra un 90 % de reutilización para verticales alternativas.
- [ ] `README.md` y `API_SPECIFICATION.md` reflejan el estado actual.
- [ ] Los ADR-01-009, ADR-01-010, ADR-01-011 y ADR-01-012 reflejan el estado final de la implementación.
- [ ] La cobertura de pruebas se verifica en >= 80 % mediante `make test-cov`.
- [ ] Los errores se registran mediante el módulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro integrante y se integra mediante _squash merge_.
- [ ] La tarea se mueve a `Completed`.
- [ ] No se registra nueva información de identificación personal (PII/PHI) en los registros.
- [ ] La ejecución pasa `ruff check .` y `ruff format .` con cero errores.
- [ ] El tipado estricto es validado por `pyright` sin errores.
- [ ] Los _pre-commit hooks_ están instalados y verificados en el entorno local.

---

## TASK-DATA-1.4-A2 — Verificar preservación del panel clasificado

> **Módulo:** DATA | **Funcionalidad:** 1.4 Contrato de Datos Final y Artefacto de Salida | **Grupo:** A Contrato de salida

### La misión

- Como Frontera de Integración,
- necesito verificar que la clasificación 1.3 preserve íntegramente el panel diario recibido,
- para que ninguna alteración silenciosa de los datos transaccionales alcance el artefacto final.

### Pasos

- Capturar el panel diario inmediatamente anterior a la clasificación.
- Comparar la identidad de las filas antes y después de ejecutar el clasificador.
- Comparar los valores de las columnas transaccionales preservadas.
- Detectar eliminaciones, adiciones, duplicaciones o reordenamientos indebidos de filas.
- Detectar modificaciones de `sku_id`, `timestamp`, `demand_qty` o `lead_time_days`.
- Permitir exclusivamente la incorporación de `sku_class`.
- Rechazar cualquier violación antes de ejecutar la persistencia.

### Criterios de aceptación

- [ ] El clasificador puede agregar `sku_class` pero no alterar las cuatro columnas provenientes del panel diario.
- [ ] La cantidad de filas permanece idéntica antes y después de clasificación.
- [ ] La identidad y los valores de cada observación permanecen invariantes.
- [ ] Una eliminación, adición, reordenamiento o mutación indebida provoca un fallo explícito.
- [ ] La comprobación ocurre antes de escribir cualquier artefacto en `processed/`.
- [ ] La implementación no recalcula ADI, CV² ni `sku_class`.
- [ ] La funcionalidad funciona según lo definido.
- [ ] El código es modular y demuestra un 90 % de reutilización para verticales alternativas.
- [ ] `README.md` y `API_SPECIFICATION.md` reflejan el estado actual.
- [ ] El ADR-01-012 documenta la preservación de filas implementada.
- [ ] La cobertura de pruebas se verifica en >= 80 % mediante `make test-cov`.
- [ ] Los errores se registran mediante el módulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro integrante y se integra mediante _squash merge_.
- [ ] La tarea se mueve a `Completed`.
- [ ] No se registra nueva información de identificación personal (PII/PHI) en los registros.
- [ ] La ejecución pasa `ruff check .` y `ruff format .` con cero errores.
- [ ] El tipado estricto es validado por `pyright` sin errores.
- [ ] Los _pre-commit hooks_ están instalados y verificados en el entorno local.

---

## TASK-DATA-1.4-B1 — Publicar artefacto final en Parquet

> **Módulo:** DATA | **Funcionalidad:** 1.4 Contrato de Datos Final y Artefacto de Salida | **Grupo:** B Persistencia columnar

### La misión

- Como Pipeline de Ingestión,
- necesito persistir el panel validado como un artefacto Apache Parquet inmutable,
- para que los Módulos 2 y 3 consuman una interfaz de datos estable y desacoplada.

### Pasos

- Recibir únicamente un panel que haya superado la validación del contrato final.
- Seleccionar y ordenar las cinco columnas definidas por el contrato de handoff.
- Normalizar los tipos finales requeridos por el esquema antes de la escritura.
- Resolver la ruta de salida a partir de la raíz de datos configurada.
- Escribir el artefacto exclusivamente dentro de `processed/` utilizando Apache Parquet.
- Evitar cualquier exportación alternativa del contrato final en formato CSV.
- Registrar mediante logging estructurado la publicación exitosa del artefacto.

### Criterios de aceptación

- [ ] El artefacto final se almacena bajo `{PRED_DATA_ROOT}/processed/`.
- [ ] El artefacto final utiliza exclusivamente formato `.parquet`.
- [ ] El Parquet contiene exactamente `sku_id`, `timestamp`, `demand_qty`, `lead_time_days` y `sku_class`.
- [ ] Los tipos persistidos corresponden al esquema estricto definido en 1.4.
- [ ] La persistencia no modifica ni sobrescribe artefactos de `raw/`.
- [ ] El escritor permanece desacoplado de la implementación interna del clasificador SBC.
- [ ] La funcionalidad funciona según lo definido.
- [ ] El código es modular y demuestra un 90 % de reutilización para verticales alternativas.
- [ ] `README.md` y `API_SPECIFICATION.md` reflejan el estado actual.
- [ ] El ADR-01-009 refleja el contrato y la ruta de publicación implementados.
- [ ] La cobertura de pruebas se verifica en >= 80 % mediante `make test-cov`.
- [ ] Los errores se registran mediante el módulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro integrante y se integra mediante _squash merge_.
- [ ] La tarea se mueve a `Completed`.
- [ ] No se registra nueva información de identificación personal (PII/PHI) en los registros.
- [ ] La ejecución pasa `ruff check .` y `ruff format .` con cero errores.
- [ ] El tipado estricto es validado por `pyright` sin errores.
- [ ] Los _pre-commit hooks_ están instalados y verificados en el entorno local.

---

## TASK-DATA-1.4-C1 — Validar handoff completo de ingesta

> **Módulo:** DATA | **Funcionalidad:** 1.4 Contrato de Datos Final y Artefacto de Salida | **Grupo:** C Integración final

### La misión

- Como Equipo de Ingeniería,
- necesito verificar el handoff completo desde el pipeline de ingesta hasta el artefacto procesado,
- para que el Módulo 1 pueda cerrarse con evidencia reproducible de integración.

### Pasos

- Ejecutar el flujo canónico utilizando la implementación real integrada de 1.3.
- Verificar el rechazo fail-closed de cualquier SKU sin demanda positiva antes de clasificación.
- Verificar la preservación del panel a través de la frontera con el clasificador.
- Leer nuevamente el Parquet producido desde `processed/`.
- Validar sobre el artefacto leído las cinco columnas y sus tipos persistidos.
- Verificar la constancia de `sku_class` por SKU.
- Ejecutar la validación con el dataset oficial utilizado durante el desarrollo del Módulo 1.

### Criterios de aceptación

- [ ] La prueba de integración utiliza la implementación real de `classify_daily_panel` y no el stub temporal.
- [ ] Un SKU sin demanda positiva detiene el pipeline antes de clasificación y persistencia.
- [ ] El artefacto leído desde Parquet satisface nuevamente el contrato estricto de 1.4.
- [ ] El flujo no duplica las implementaciones de ADI, CV² o clasificación SBC pertenecientes a 1.3.
- [ ] El dataset de validación produce 4862 filas y 10 SKU en el artefacto final.
- [ ] Los 10 SKU del dataset de validación conservan la clasificación `intermittent` obtenida por 1.3.
- [ ] La ejecución completa demuestra la frontera `1.2 → 1.3 → 1.4 → processed/`.
- [ ] La funcionalidad funciona según lo definido.
- [ ] El código es modular y demuestra un 90 % de reutilización para verticales alternativas.
- [ ] `README.md` y `API_SPECIFICATION.md` reflejan el estado actual.
- [ ] Los ADR-01-009, ADR-01-010, ADR-01-011 y ADR-01-012 están sincronizados con la implementación definitiva.
- [ ] La cobertura de pruebas se verifica en >= 80 % mediante `make test-cov`.
- [ ] Los errores se registran mediante el módulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro integrante y se integra mediante _squash merge_.
- [ ] La tarea se mueve a `Completed`.
- [ ] No se registra nueva información de identificación personal (PII/PHI) en los registros.
- [ ] La ejecución pasa `ruff check .` y `ruff format .` con cero errores.
- [ ] El tipado estricto es validado por `pyright` sin errores.
- [ ] Los _pre-commit hooks_ están instalados y verificados en el entorno local.
