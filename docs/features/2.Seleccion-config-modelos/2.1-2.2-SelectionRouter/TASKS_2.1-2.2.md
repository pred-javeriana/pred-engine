# Tareas 2.1 / 2.2 — compiladas en esta sesion

Fuente Notion: `project/current-context/target_task.md`.
Codigo: `project/current-plan/IMPLEMENTATION.md`.

Implementacion completada en `src/pred_engine/optimizacion/router/` y
`tests/optimizacion/router/` (Steps 1–6 de `IMPLEMENTATION.md`).
Verificacion: 387 pruebas, cobertura global >= 80 %.

Orden de compilacion (dependencias reales, no el orden lineal de Notion):

1. 2.1-A1 contratos de seleccion
2. 2.2-A1 contratos de politica (el router los necesita)
3. 2.1-A2 registro
4. 2.2-A2 politica inicial
5. 2.1-B1 enrutador
6. 2.2-B1 integracion + cobertura
7. documentacion (este directorio + ADR-008/009)

---

## TASK-SEL-2.1-A1 — Definir contratos comunes de seleccion

> **Modulo:** SEL | **Funcionalidad:** 2.1 Arquitectura del Motor de Seleccion y Optimizacion | **Grupo:** A Contratos y estrategias

### La mision

- Como motor de seleccion de PRED,
- necesito definir contratos comunes de entrada, estrategia y resultado,
- para que el enrutador pueda delegar trabajo sin depender de implementaciones concretas.

### Pasos

- Definir el contrato tipado de entrada requerido por el motor de seleccion.
- Definir el contrato comun que deben satisfacer las estrategias de seleccion.
- Definir el contrato comun de resultado producido por una estrategia.
- Representar explicitamente las familias de predictores soportadas.
- Mantener los contratos independientes de Optuna, Walk-Forward y modelos concretos.

### Criterios de aceptacion

- [x] Definir contratos suficientes para invocar estrategias sin conocer su implementacion concreta.
- [x] Evitar dependencias directas hacia Optuna, ASHA, Walk-Forward o modelos especificos.
- [x] Representar explicitamente las familias soportadas y rechazar valores invalidos.
- [x] Verificar los contratos mediante pruebas unitarias.
- [x] README.md y API_SPECIFICATION.md reflejan el estado actual.
- [x] La cobertura de pruebas se verifica en >= 80 %.
- [x] Los errores se registran mediante el modulo `logging` de Python (`logger`).

**Compilado en:** `IMPLEMENTATION.md` Step 1.
**Archivos:** `router/contratos.py`, `router/errores.py`, `tests/.../test_contratos.py`.

---

## TASK-SEL-2.1-A2 — Implementar registro desacoplado de estrategias

> **Modulo:** SEL | **Funcionalidad:** 2.1 | **Grupo:** A Contratos y estrategias

### Pasos

- Implementar un registro de estrategias indexado por familia.
- Permitir registrar implementaciones que satisfagan el contrato comun.
- Implementar la resolucion determinista de una estrategia registrada.
- Rechazar explicitamente familias sin estrategia disponible.
- Mantener el registro independiente de la politica topológica.

**Compilado en:** Step 3. `router/registro.py`, `tests/.../test_registro.py`.

---

## TASK-SEL-2.1-B1 — Implementar enrutador de seleccion sin estado

> **Modulo:** SEL | **Funcionalidad:** 2.1 | **Grupo:** B Orquestacion del enrutador

### Pasos

- Implementar `SelectionRouter` sobre los contratos definidos.
- Recibir una entrada tipada sin realizar lectura directa de archivos.
- Consultar la politica de enrutamiento para obtener las decisiones aplicables.
- Resolver cada estrategia mediante el registro.
- Delegar la ejecucion sin implementar logica interna de seleccion.
- Retornar los resultados utilizando el contrato comun.

**Compilado en:** Step 5. `router/enrutador.py`.
**Nota:** se compiló despues de 2.2-A1 porque consulta `RoutingPolicy`.

---

## TASK-SEL-2.2-A1 — Modelar politica y decision de enrutamiento

> **Modulo:** SEL | **Funcionalidad:** 2.2 Enrutamiento basado en la clasificacion del SKU | **Grupo:** A Politica topológica

### Pasos

- Definir el contrato de `RoutingPolicy`.
- Definir el contrato de `RoutingDecision`.
- Representar en cada decision la familia candidata y el perfil topológico.
- Representar los perfiles `dense_stable`, `dense_variable`, `sparse_stable` y `sparse_variable`.
- Mantener los contratos independientes de modelos e hiperparametros concretos.

**Compilado en:** Step 2 (antes del router).

---

## TASK-SEL-2.2-A2 — Implementar politica topológica inicial

> **Modulo:** SEL | **Funcionalidad:** 2.2 | **Grupo:** A Politica topológica

### Pasos

- Implementar la politica para `smooth`, `erratic`, `intermittent`, `lumpy`.
- Asociar a cada clase su perfil topológico.
- Rechazar clases desconocidas sin valores predeterminados silenciosos.

**Compilado en:** Step 4. `router/politica.py`.
**Literales:** minusculos del contrato 1.4, no Title Case de Notion.

---

## TASK-SEL-2.2-B1 — Integrar politica con enrutador de seleccion

> **Modulo:** SEL | **Funcionalidad:** 2.2 | **Grupo:** B Integracion del enrutamiento

### Pasos

- Inyectar la politica de enrutamiento en el `SelectionRouter`.
- Resolver las decisiones correspondientes al `sku_class` recibido.
- Resolver una estrategia registrada por cada familia candidata.
- Propagar el perfil topológico hacia la estrategia.
- Evitar ejecutar estrategias excluidas por la politica.
- Registrar las decisiones de enrutamiento tomadas para cada SKU.

**Compilado en:** Step 6. Pruebas parametrizadas + AST de desacoplamiento + omit de coverage.
