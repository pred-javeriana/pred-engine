# Tareas 2.4 — HPO (TPE Sampler + ASHA Resource Allocator)

Fuente: Notion, pagina "2.4 HPO" (hija de "2. Seleccion de configuracion y
modelo"), 9 tareas unicas asignadas a Tomas Ramirez Roa (`TASK-HPO-4.0-*`,
`TASK-HPO-5.0-*`), todas `P1-Important`, estado `Backlog` al momento de
escribir esto.

**Nota de limpieza en Notion (pendiente, gestion no codigo):** existe un
duplicado exacto de `TASK-HPO-4.0-A2` (dos paginas identicas, userDefined:ID
40 y 48) — fusionar/eliminar una antes de mover tareas a `Completada`.

**Hallazgo que enmarca todo lo demas:** el motor de HPO (TPE + ASHA +
Walk-Forward) ya existia, construido en sesiones anteriores bajo otra
numeracion de tareas (`TASK-SEL-2.81-*`, `TASK-SEL-2.9-*`, ya mergeadas).
Esta sesion no reconstruye ese motor: reconcilia Notion/ADRs con el, y
cierra las brechas funcionales genuinamente ausentes. Ver `README.md` y
ADR-012/ADR-013.

---

## TASK-HPO-4.0-A1 — Diseñar arquitectura del TPE Sampler

**Estado: satisfecho por codigo preexistente + documentacion de esta sesion.**

La interfaz publica (`initialize`/`suggest`/`update`/`get_history` de
Notion ≈ `ask`/`suggest_*`/`tell`/`trials_finalizados` de `contratos.py`),
la estructura de historial (config → metrica, `InfoTrial`) y el manejo de
espacios mixtos ya existian en `espacio.py`/`contratos.py`/`muestreadores.py`.

Cerrado en esta sesion:

- [x] Tipo `Ordinal` (codificacion de ordinales como enteros ordenados,
  criterio explicito de Notion que faltaba).
- [x] ADR-012 formaliza los supuestos de TPE y por que se delega a Optuna.
- [x] Diagrama (Mermaid, en `README.md`) sincronizado con `contratos.py`/`espacio.py`.
- [x] `API_SPECIFICATION.md` documenta la interfaz publica.
- [ ] "Documentacion API autogenerada": no hay `pdoc`/`sphinx` configurado
  en el repo hoy; se documenta manualmente via README/API_SPECIFICATION
  (mismo patron que 2.1-2.2 y 2.9). Flag para el equipo si se quiere una
  herramienta de generacion automatica en el futuro.

**Archivos:** `espacio.py`, `contratos.py`, `muestreadores.py`,
`docs/adr/ADR-012-tpe-sobre-optuna.md`.

---

## TASK-HPO-4.0-A2 — Implementar TPE (separacion buena/mala y estimacion de densidades)

**Estado: la separacion buena/mala y densidades l(θ)/g(θ) ya funcionan (delegadas
a `optuna.samplers.TPESampler`, ADR-012). Warm start era una brecha real; cerrada
en esta sesion.**

- [x] TPE produce propuestas validas (ya probado en `test_muestreadores.py`).
- [x] Espacios mixtos/jerarquicos correctos (`espacio.py`, sin asumir
  independencia problematica mas alla de la propia de TPE, documentada en
  ADR-012).
- [x] **Warm start para SKUs lumpy**: `semillas_desde_historico` /
  `inyectar_historico` (`adaptador_optuna.py`), expuesto como
  `EstudioHPO.agregar_trials_historicos`; `ejecutar_estudio(...,
  historico_previo=...)`. Tests: `test_warm_start_*` en
  `test_estudio.py`/`test_adaptador_optuna.py`.
- [x] Cobertura >= 80 % (paquete completo: 92 %, ver README).
- [x] Logging via `pred_engine.comun.logger`.

**Archivos:** `adaptador_optuna.py`, `estudio.py`,
`tests/optimizacion/HPO/test_adaptador_optuna.py`,
`tests/optimizacion/HPO/test_estudio.py`.

---

## TASK-HPO-4.0-B1 — Implementar gestion de historial y persistencia

**Estado: satisfecho por codigo preexistente (2.9) + timestamp de esta sesion.**

- [x] Historial serializable/deserializable sin perdida (`volcar_jsonl` /
  `reanudar_estudio`, ya probado en 2.9).
- [x] Formato `{config, metric_value, timestamp, status}`: `timestamp`
  agregado en esta sesion (`Trial.timestamp`, `trial.set_user_attr("timestamp",
  ...)` en `estudio.py`).
- [x] Checkpoints permiten reanudar (`ControladorReanudacion`, ADR-010/011).
- [x] Validacion de consistencia (`IncompatibilidadCorridaError`,
  `ManifiestoCorruptoError`).
- [x] Integracion con ASHA: el registro de historial es el mismo estudio
  que ASHA consulta via `PodadorASHAOptuna` (no hay dos fuentes de verdad).
- [~] Sincronizacion del muestreador al reanudar: cubierta para
  `RandomSampler` (2.9); para `TPESampler` se documenta como limitacion
  conocida y se agrega una advertencia explicita (antes silenciosa) — ver
  ADR-012 y `test_reanudacion_con_tpe_preserva_los_trials_ya_evaluados`.

**Archivos:** `adaptador_optuna.py`, `registro.py`,
`comun/dataclasses/hpo.py`, `tests/optimizacion/HPO/test_estudio_reanudacion.py`.

---

## TASK-HPO-4.0-C1 — Escribir pruebas de regresion y benchmarking del TPE

**Estado: brecha real, cerrada en esta sesion (no existia ningun benchmark
sphere/Rastrigin/Rosenbrock en el repo antes).**

- [x] Funciones objetivo sphere/Rastrigin/Rosenbrock (`benchmarks/funciones_objetivo.py`).
- [x] TPE converge mas cerca del minimo global que random search en las 3
  funciones objetivo (>= 3 exigido; estable en 4 semillas probadas).
  GP-BO/grid search no se incluyen en el comparador automatizado: Optuna no
  trae GP-BO sin un plugin adicional, y `GridSampler` no es comparable con
  un espacio continuo sin discretizar — se documenta como limitacion en vez
  de forzar una comparacion poco representativa.
- [x] Tests de regresion parametrizados (`test_convergencia_tpe.py`,
  marcados `@pytest.mark.slow`).
- [x] Resultados documentados en `README.md`.

**Archivos:** `benchmarks/funciones_objetivo.py`,
`benchmarks/comparador_convergencia.py`,
`tests/optimizacion/HPO/benchmarks/`.

---

## TASK-HPO-5.0-A1 — Diseñar arquitectura del HPO Resource Allocator (ASHA)

**Estado: satisfecho por codigo preexistente + documentacion de esta sesion.**

La interfaz de decision (`should_prune`/`get_pruned_configs` de Notion ≈
`DecisorASHA.decidir`/`Trial.motivo` de podados), el registro de
configuraciones evaluadas y las reglas de seguridad de poda ya existian en
`asha.py`/`poda.py`.

- [x] Reglas de poda documentadas explicitamente (ADR-013).
- [x] Criterio de "promesa" (ranking por escalon) definido formalmente
  (antes vivia implicito en `DecisorASHA.decidir`) — ver
  `API_SPECIFICATION.md`.
- [x] Diagrama de flujo (Mermaid, README) sincronizado con el codigo.
- [x] **Reconciliacion explicita** con la redaccion de Notion ("percentil
  75 de todas las configuraciones" vs. "mejor 1/factor_reduccion del mismo
  escalon"): documentada como decision deliberada en ADR-013, no como
  discrepancia a corregir.
- [ ] "Documentacion API autogenerada": mismo flag que 4.0-A1.

**Archivos:** `asha.py`, `poda.py`,
`docs/adr/ADR-013-asha-secuencial-sin-multiworker.md`.

---

## TASK-HPO-5.0-A2 — Implementar registro compartido y gestion de workers

**Estado: redefinido de alcance (decision de arquitectura, ADR-013), no
implementado literalmente.**

Los criterios originales (dict/SQLite con `worker_id`, locks de
concurrencia, tests con multiples *workers* simulados) asumen un sistema
multi-proceso/multi-hilo evaluando trials de UN mismo estudio en paralelo.
La arquitectura real es un bucle secuencial de un solo proceso; el
paralelismo del repo ocurre a nivel de SKU
(`classical_selection.py::seleccionar_por_panel`, `ProcessPoolExecutor`,
estudios completamente aislados entre si — sin necesidad de un registro
compartido).

- [x] "Registro" real para el modo secuencial: `control_reanudacion` +
  `backend.jsonl` (2.9) ya cumplen "orden, quien evalua, estado".
- [x] Documentado explicitamente como fuera de alcance de esta version en
  ADR-013 y `README.md` ("Fuera de alcance"), no como deuda silenciosa.
- [ ] Registro thread-safe con locks / `ThreadPoolExecutor`: **no
  implementado a proposito** — construirlo sin un consumidor real seria
  codigo no ejercitado. Revisar ADR-013 si aparece un caso de uso real de
  paralelizar trials de un mismo estudio.

**Decision que requiere reconciliar Notion:** marcar esta tarea segun el
alcance redefinido (no "Completada" en el sentido literal original, sino
"Redefinida/Documentada" o equivalente segun el flujo del equipo).

---

## TASK-HPO-5.0-B1 — Implementar poda greedy con reglas de seguridad

**Estado: satisfecho por codigo preexistente + tests adicionales de esta sesion.**

- [x] Regla de >= 4 ventanas antes de podar (`ReglasPoda.min_ventanas`,
  probado en `test_asha.py`).
- [x] Metrica robusta a outliers (`agregacion="media_recortada"` por
  defecto).
- [x] Poda semantica detecta modelos degenerados (`es_degenerada`).
- [x] Registro de podas completo (`motivo` obligatorio en cada poda).
- [x] Tests con series simuladas adicionales (lumpy, erratico) agregados
  en esta sesion en `test_poda.py`/`test_asha.py`.

**Archivos:** `poda.py`, `asha.py`, `tests/optimizacion/HPO/test_poda.py`,
`tests/optimizacion/HPO/test_asha.py`.

---

## TASK-HPO-5.0-B2 — Implementar asincronia con workers

**Estado: redefinido de alcance, analogo a 5.0-A2 (ADR-013).**

La "asincronia" de ASHA ya implementada es **algoritmica** (regla de
seguridad #1/#2: cada trial se compara contra competidores del mismo
escalon sin esperar una ronda completa), no concurrencia literal de
hilos/procesos. `allocate_work_to_worker()`/`report_evaluation_result()`
con backoff/retry no se implementan por la misma razon que en 5.0-A2: sin
un consumidor multi-worker real, seria infraestructura no ejercitada.

- [x] Poda automatica tras cada reporte: ya ocurre en `_correr_trial`
  (`estudio.py`) — `trial.should_prune()` se consulta despues de cada
  `trial.report()`.
- [x] Documentado como fuera de alcance en ADR-013.
- [ ] `ThreadPoolExecutor`/backoff/retry: no implementado a proposito.

---

## TASK-HPO-5.0-C1 — Escribir pruebas de convergencia y auditoria de ASHA

**Estado: brecha real (parcial), cerrada en esta sesion.**

Ya existia `test_poda_reduce_las_ventanas_evaluadas_respecto_al_modo_sin_poda`
pero solo verificaba una desigualdad estricta, no el umbral cuantitativo.

- [x] ASHA reduce el costo de evaluacion en >= 30 % vs. sin poda (medido:
  48.3 % con la configuracion de prueba; proxy = ventanas Walk-Forward
  evaluadas, no tiempo de reloj — ver justificacion en el test y en
  `README.md`).
- [x] Todas las configuraciones podadas tienen `motivo` no vacio (registro
  de auditoria completo): `test_todos_los_trials_podados_tienen_motivo_auditable`.
- [x] Tests de regresion de eficiencia (`test_asha_reduce_el_costo_de_evaluacion_en_al_menos_30_por_ciento`).
- [x] Resultados documentados en `README.md`.

**Archivos:** `tests/optimizacion/HPO/test_estudio.py`.
