# ADR-013: ASHA como decisor puro, ejecucion secuencial de un proceso

**Date:** 2026-09-21
**Status:** Accepted
**Notion ADR:** ADR-02-005 (estado en Notion: actualizar de "Propuesto" a
"Aceptada" — la decision ya esta implementada y probada; ver Consequences)
**Notion Task:** TASK-HPO-5.0-A1 / A2 / B1 / B2

## Context

TPE decide **que** configuracion evaluar; no decide **cuantos recursos**
(ventanas Walk-Forward) dedicarle antes de descartarla. Notion (ADR-02-005)
propone Asynchronous Successive Halving (ASHA) para eso, con un registro
compartido consultado por multiples *workers* concurrentes que toman la
configuracion mas prometedora disponible sin esperar una ronda completa.

Al momento de este ADR, `asha.py::DecisorASHA` + `poda.py::ReglasPoda` ya
implementan el algoritmo de decision, y `estudio.py::ejecutar_estudio` ya lo
integra -- pero como un **bucle secuencial de un solo proceso**
(`while indice_trial < n_trials: trial = study.ask(); ...`), no como un
sistema con multiples workers, locks y un registro compartido thread-safe.

## Decision

Se adopta la implementacion existente como version definitiva del Modulo
2.4 para este resource allocator:

1. **`DecisorASHA` es un algoritmo puro** (`dict[int, float]` de entrada,
   `DecisionPoda` de salida): no conoce Optuna ni un `study` concreto. El
   adaptador (`PodadorASHAOptuna` en `adaptador_optuna.py`) es quien lo
   conecta a `optuna.pruners.BasePruner`.
2. **Reglas de seguridad (`ReglasPoda`), ya implementadas:**
   - `min_ventanas=4`: ninguna poda antes de completar 4 ventanas.
   - `agregacion="media_recortada"` (`proporcion_recorte=0.1`): la metrica
     de comparacion es una agregacion robusta de las ventanas evaluadas, no
     una ventana aislada.
   - `habilitar_poda_semantica=True` (`poda.py::es_degenerada`): descarta
     configuraciones que producen predicciones no finitas, negativas, o
     nulas cuando el entrenamiento no lo era.
   - Toda poda registra `motivo` (config, escalon, valor, motivo) --
     `Trial.motivo` nunca es `None` para un trial podado (auditable).
3. **Criterio de "promesa" (orden de merito):** en cada escalon geometrico
   (`min_ventanas`, `min_ventanas*factor_reduccion`, ...), un trial se poda
   si su valor esta por debajo del mejor `1/factor_reduccion` de sus
   **competidores que llegaron al mismo escalon** -- no contra "todas las
   configuraciones" sin importar cuantas ventanas llevan. Esto difiere de
   la redaccion literal de Notion ("descartar si `metric >
   percentil_75_de_todas_las_configs`"): comparar solo contra el mismo
   escalon es la semantica estandar de ASHA (Li et al. 2018) y evita
   penalizar a un trial joven contra uno que ya evaluo mas ventanas.
4. **Ejecucion secuencial de un proceso**, no multi-worker: se documenta
   como alcance explicito de esta version (ver Alternatives Considered),
   no como una brecha a cerrar. El registro de "quien evalua que" es
   `control_reanudacion` + `backend.jsonl` (ver ADR-010/011), suficiente
   para el modo secuencial de hoy.

## Rationale

- Las 4 reglas de seguridad ya cierran el riesgo que motivo ADR-02-005
  (podar por ruido de corto plazo) sin necesitar concurrencia real.
- `seleccionar_por_panel` (unico consumidor de paralelismo en el repo hoy,
  en `classical_selection.py`) ya paraleliza a nivel de **SKU** con
  `ProcessPoolExecutor` -- cada proceso corre su propio estudio secuencial
  aislado. No hay un caso de uso hoy que requiera paralelizar *trials de un
  mismo estudio* con un registro compartido entre ellos.
- Construir un registro compartido thread-safe y un asignador de workers
  sin un consumidor real seria codigo no ejercitado en produccion --
  precisamente lo que el equipo decidio evitar al cerrar este modulo sobre
  la arquitectura ya construida en vez de reconstruir desde cero.

## Consequences

- **Accion en Notion:** actualizar el estado de ADR-02-005 de "Propuesto" a
  "Aceptada" (o "Aplicada"), y reconciliar la redaccion de los criterios de
  aceptacion de TASK-HPO-5.0-A2 y TASK-HPO-5.0-B2 (registro compartido con
  `worker_id`/locks, `allocate_work_to_worker()`, `ThreadPoolExecutor`) para
  reflejar que quedan **fuera de alcance de esta version** por decision
  arquitectonica, no pendientes de implementar. Ver
  `docs/features/2.Seleccion-config-modelos/2.4-HPO/API_SPECIFICATION.md`.
- Si en el futuro aparece un consumidor real que necesite paralelizar
  trials de UN estudio (no SKUs), este ADR es el punto de partida a
  revisar -- no un veto permanente.
- `factor_reduccion` (no un `percentil_corte` literal) sigue siendo el
  parametro de `ReglasPoda` que controla la agresividad de la poda.

## Alternatives Considered

- **Registro compartido thread-safe + `ThreadPoolExecutor` (literal a
  Notion):** rechazado para esta version. Sin un consumidor real, seria
  infraestructura no ejercitada (dead code en la practica) que contradice
  el criterio de "codigo modular... reutilizacion", que asume uso real, no
  hipotetico.
- **`percentil_corte=75.0` explicito en `ReglasPoda` para calzar
  literalmente con Notion:** rechazado. Cambiaria el comportamiento de poda
  ya probado (`factor_reduccion=3` ≈ retener el mejor tercio del escalon)
  sin una razon funcional -- solo por igualar una redaccion. Se documenta
  la equivalencia semantica en vez de forzar el cambio.
- **Poda determinista sincrona (Successive Halving simple, Option A de
  Notion):** rechazada por el propio ADR-02-005 original: crea cuellos de
  botella con *workers* rapidos esperando al mas lento del grupo.
