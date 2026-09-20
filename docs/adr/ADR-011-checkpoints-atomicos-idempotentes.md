# ADR-011: Checkpoints atomicos, huella e idempotencia en frontera de trial

**Date:** 2026-09-20
**Status:** Accepted
**Notion ADR:** ADR-02-014
**Notion Task:** TASK-SEL-2.9-A2 / B1 / C2

## Context

Reanudar un estudio HPO exige:

1. No corromper el ultimo estado valido si la escritura se interrumpe.
2. Rechazar silenciosamente configuraciones distintas sobre el mismo
   `run_id`.
3. Permitir recuperaciones repetidas sin duplicar trials ni violar la
   maquina de estados.
4. Definir la granularidad de recuperacion cuando un trial queda activo
   a mitad de walk-forward.

Notion ADR-02-014 agrupa huella determinista, escritura atomica,
idempotencia de checkpoints y frontera de trial.

## Decision

### Huella determinista

- `calcular_huella(SolicitudCorrida)` → SHA-256 sobre payload canonico
  JSON ordenado.
- Participan: familia, SKU, seed, metrica, validacion, espacio canonico,
  reglas de optimizador, `n_observaciones`, `huella_serie`, backend.
- `run_id` **no** entra en la huella (identifica la corrida, no el
  experimento).
- `huella_serie` = SHA-256 de `y` en float64; evita reanudar sobre otra
  serie de la misma longitud.
- Rechazo → `IncompatibilidadCorridaError(motivos)` **sin** mutar el
  manifiesto persistido.

### Escritura atomica

- `escribir_atomico`: `archivo.tmp` + `os.replace`.
- Aplica a `manifiesto.json` y a `backend.jsonl`.
- Un manifiesto o JSONL a medias nunca reemplaza al ultimo valido.

### Idempotencia

- Transicion identidad `en_progreso` → `en_progreso` permitida para
  checkpoints repetidos.
- `completada` → `completada` permitida al reconstruir sin reejecutar.
- `abrir` sobre corrida completada compatible devuelve handle restaurado
  sin transicionar a `en_progreso`.

### Frontera de trial

- Checkpoint en `ejecutar_estudio` **despues** de cada trial finalizado
  (`tell` completado / podado / fallido).
- Trial `RUNNING` (interrupcion intra-walk-forward) **no** se serializa.
  Al reanudar se reejecuta ese trial desde cero.
- **No** hay checkpoint intra-ventana.
- `EN_PROGRESO` es reanudable aunque no se haya escrito `interrumpida`
  (recuperacion no senalizada tras kill -9).

### Sincronizacion del muestreador (RandomSampler)

Tras restaurar trials con `add_trial`, un `RandomSampler` nuevo repite la
primera muestra del stream si no se avanza el RNG. `_sincronizar_sampler_aleatorio`
consume `n_trials_cargados` muestras antes del primer `ask()` post-restore
para alinear la secuencia con una corrida continua.

## Rationale

- La huella fail-closed evita mezclar trials de configuraciones distintas
  bajo el mismo `run_id`.
- Atomico en ambos artefactos cierra la ventana donde un volcado JSONL
  parcial corromperia la fuente de verdad de Optuna.
- La frontera de trial es el compromiso entre granularidad util y
  complejidad de serializar estados parciales de `EjecutorGreedy`.
- Idempotencia hace seguros reintentos de recuperacion y checkpoints
  duplicados.

## Consequences

- Prueba integral: interrupcion simulada vs corrida continua debe producir
  el mismo conjunto de trials finalizados (`test_estudio_reanudacion.py`).
- Cambiar `y`, seed, espacio o reglas de poda exige nuevo `run_id` o
  corrida nueva.
- TPE u otros muestreadores no tienen sincronizacion automatica hoy; solo
  `RandomSampler` esta cubierto en `reanudar_estudio`.
- `schema_version` en manifiesto permite migraciones futuras; version
  distinta → `VersionManifiestoError`.

## Alternatives Considered

- **Reanudacion intra-ventana:** rechazado; serializar estado de
  `EjecutorGreedy` y metricas parciales multiplica superficie y no estaba
  en el AC de 2.9.
- **Persistir RUNNING como FAIL:** rechazado; ensucia el estudio y no
  garantiza reproducibilidad ASHA sin reejecutar igualmente.
- **Huella sin serie:** rechazado; misma longitud con valores distintos
  pasaria compatibilidad.
- **Solo reanudar `interrumpida`:** rechazado; kill -9 dejaria corridas
  irrecuperables.
