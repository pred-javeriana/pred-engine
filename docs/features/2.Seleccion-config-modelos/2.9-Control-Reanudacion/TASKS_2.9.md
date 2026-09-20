# Tareas 2.9 — compiladas en esta sesion

Fuente Notion: `project/current-context/target_task.md`.
Codigo: `project/current-plan/IMPLEMENTATION.md`.

Implementacion completada en `src/pred_engine/optimizacion/control_reanudacion/`,
`src/pred_engine/optimizacion/optimizadores/HPO/adaptador_optuna.py`,
`src/pred_engine/optimizacion/optimizadores/HPO/estudio.py` y
`tests/optimizacion/control_reanudacion/` (Steps 1–6 de `IMPLEMENTATION.md`).
Verificacion: 437 pruebas, cobertura global >= 80 %.

Orden de compilacion (dependencias reales, no el orden lineal de Notion):

1. 2.9-A1 manifiesto y estados
2. 2.9-A2 huella y compatibilidad
3. 2.9-B1 almacenamiento atomico
4. 2.9-B2 puerto del motor + adaptador Optuna
5. 2.9-C1 controlador
6. 2.9-C2 integracion en `ejecutar_estudio` + cobertura
7. documentacion (este directorio + ADR-010/011)

---

## TASK-SEL-2.9-A1 — Definir manifiesto y estados de corrida

> **Modulo:** SEL | **Funcionalidad:** 2.9 Control de reanudacion y reproducibilidad | **Grupo:** A Estado y compatibilidad

### La mision

- Como controlador de reanudacion,
- necesito representar explicitamente una corrida y su ciclo de vida,
- para que las interrupciones y continuaciones puedan gestionarse de forma reproducible.

### Pasos

- Definir el contrato del manifiesto de corrida.
- Representar los estados nueva, en progreso, interrumpida, completada y fallida.
- Incluir identificador de corrida, familia, SKU, semilla y metrica objetivo.
- Incluir configuracion de validacion y referencia al punto de control del motor.
- Incluir una version explicita del esquema del manifiesto.
- Validar transiciones permitidas entre estados.

### Criterios de aceptacion

- [x] Representar todos los metadatos obligatorios definidos en la seccion 2.9.
- [x] Rechazar transiciones invalidas del ciclo de vida.
- [x] Mantener el manifiesto independiente de Optuna.
- [x] Verificar serializacion y reconstruccion sin perdida de informacion.
- [x] README.md y API_SPECIFICATION.md reflejan el estado actual.
- [x] La cobertura de pruebas se verifica en >= 80 %.
- [x] Los errores se registran mediante el modulo `logging` de Python (`logger`).

**Compilado en:** `IMPLEMENTATION.md` Step 1.
**Archivos:** `control_reanudacion/contratos.py`, `errores.py`, `transiciones.py`,
`tests/.../test_contratos.py`, `test_transiciones.py`.

---

## TASK-SEL-2.9-A2 — Implementar huella y compatibilidad de corrida

> **Modulo:** SEL | **Funcionalidad:** 2.9 | **Grupo:** A Estado y compatibilidad

### La mision

- Como controlador de reanudacion,
- necesito identificar deterministicamente la configuracion experimental de una corrida,
- para que un punto de control incompatible nunca sea reutilizado silenciosamente.

### Pasos

- Definir los campos que participan en la huella de configuracion.
- Normalizar la representacion de la configuracion antes de calcular la huella.
- Calcular una huella determinista para configuraciones equivalentes.
- Comparar la solicitud actual con el manifiesto almacenado.
- Rechazar diferencias en semilla, metrica, validacion, espacio de busqueda o configuracion relevante.
- Registrar explicitamente los motivos de incompatibilidad.

**Compilado en:** Step 2. `control_reanudacion/huella.py`, `optimizadores/HPO/espacio.py`
(`descripcion_canonica`), `tests/.../test_huella.py`.

### Criterios de aceptacion

- [x] Producir la misma huella para configuraciones semanticamente identicas.
- [x] Producir huellas distintas cuando cambie un parametro relevante.
- [x] Rechazar una reanudacion cuando la huella no coincida.
- [x] Informar que validacion de compatibilidad impidio la reanudacion.

---

## TASK-SEL-2.9-B1 — Implementar almacenamiento atomico de manifiestos

> **Modulo:** SEL | **Funcionalidad:** 2.9 | **Grupo:** B Persistencia segura

### La mision

- Como controlador de reanudacion,
- necesito persistir manifiestos sin corromper el ultimo estado valido,
- para que una interrupcion durante la escritura no destruya la capacidad de recuperacion.

**Compilado en:** Step 3. `control_reanudacion/almacenamiento.py`,
`tests/.../test_almacenamiento.py`.

### Criterios de aceptacion

- [x] Mantener intacto el ultimo manifiesto valido ante una escritura interrumpida simulada.
- [x] Reconstruir correctamente un manifiesto persistido.
- [x] Rechazar manifiestos corruptos o con version incompatible.
- [x] Mantener la capa de almacenamiento independiente del motor HPO.

---

## TASK-SEL-2.9-B2 — Adaptar puntos de control del motor HPO

> **Modulo:** SEL | **Funcionalidad:** 2.9 | **Grupo:** B Persistencia segura

### La mision

- Como controlador de reanudacion,
- necesito delegar la persistencia especifica del motor HPO mediante un contrato estable,
- para que el componente generico no dependa directamente de Optuna.

**Compilado en:** Step 4. `AdaptadorPersistenciaOptuna`, `volcar_jsonl` (filtra
RUNNING), `reanudar_estudio`, `tests/optimizacion/HPO/test_adaptador_optuna.py`.

### Criterios de aceptacion

- [x] Persistir y reconstruir un estudio mediante el contrato del motor.
- [x] Preservar los estados completado, podado y fallido de los ensayos.
- [x] Mantener el componente generico sin importaciones de Optuna.
- [x] Reutilizar la persistencia existente sin crear un segundo registro paralelo de ensayos.

---

## TASK-SEL-2.9-C1 — Implementar controlador de reanudacion

> **Modulo:** SEL | **Funcionalidad:** 2.9 | **Grupo:** C Orquestacion de recuperacion

### La mision

- Como motor de ejecucion de PRED,
- necesito un controlador que decida entre crear, continuar o rechazar una corrida,
- para que la recuperacion sea segura, trazable e idempotente.

**Compilado en:** Step 5. `control_reanudacion/controlador.py`,
`tests/.../test_controlador.py`.

### Criterios de aceptacion

- [x] Crear correctamente una corrida nueva cuando no exista estado previo.
- [x] Reanudar una corrida interrumpida cuando su configuracion sea compatible.
- [x] Rechazar una corrida incompatible sin modificar su estado persistido.
- [x] No volver a ejecutar automaticamente una corrida marcada como completada.
- [x] Mantener comportamiento idempotente ante recuperaciones repetidas.

---

## TASK-SEL-2.9-C2 — Integrar reanudacion con el motor HPO

> **Modulo:** SEL | **Funcionalidad:** 2.9 | **Grupo:** C Orquestacion de recuperacion

### La mision

- Como motor HPO de PRED,
- necesito utilizar el controlador de reanudacion al ejecutar un estudio,
- para que una interrupcion conserve los ensayos finalizados y permita continuar el trabajo pendiente.

**Compilado en:** Step 6. `estudio.py` (kwargs `raiz_corrida`, `run_id`,
`controlador`), `tests/optimizacion/HPO/test_estudio_reanudacion.py`,
`pyproject.toml` (coverage omit, pyright, pythonpath).

### Criterios de aceptacion

- [x] Continuar un estudio recuperado sin repetir ensayos previamente finalizados.
- [x] Permitir reejecutar desde el inicio unicamente el ensayo que estuviera incompleto.
- [x] Preservar resultados, estados y metricas intermedias de los ensayos restaurados.
- [x] Evitar ensayos duplicados ante multiples intentos de recuperacion.
- [x] Marcar correctamente la corrida como completada al finalizar.
- [x] Verificar mediante una prueba integral que una corrida interrumpida produce el
  mismo conjunto de ensayos finalizados que una ejecucion equivalente sin interrupcion.
