# ADR-010: Control de corrida generico vs backend HPO (Option B)

**Date:** 2026-09-20
**Status:** Accepted
**Notion ADR:** ADR-02-013
**Notion Task:** TASK-SEL-2.9-B2 / C1 / C2

## Context

El Modulo 2 necesita reanudar estudios HPO tras interrupciones sin perder
trials finalizados ni mezclar configuraciones incompatibles. Optuna ya
expone `volcar_jsonl` / `reanudar_estudio` en `adaptador_optuna.py`; el
motor generico (`registro.py`, `estudio.py`) traduce trials a
`ResultadoEstudio` sin importar el backend.

Si el control de reanudacion importara `optuna.Study` o duplicara el
registro de trials, el paquete generico quedaria acoplado a un vendor y
habria dos fuentes de verdad para el mismo estudio.

Notion ADR-02-013 plantea dos opciones: (A) persistencia monolitica en el
controlador, o (B) manifiesto PRED + referencia opaca al checkpoint del
motor.

## Decision

Se adopta **Option B: manifiesto + puerto del motor**.

- `control_reanudacion/` define `ManifiestoCorrida`, huella, estados y
  `ControladorReanudacion`. **No importa Optuna.**
- `PuertoPersistenciaMotor` es un `Protocol` con `crear`, `persistir`,
  `restaurar`. El `handle` es opaco para el controlador.
- `AdaptadorPersistenciaOptuna` es el unico modulo que serializa trials;
  reutiliza `volcar_jsonl` / `reanudar_estudio` existentes (ADR-02-010 /
  JSONL como formato de checkpoint).
- El manifiesto guarda `backend_checkpoint` (nombre de archivo, p. ej.
  `backend.jsonl`), no el contenido de los trials.
- `registro.py` sigue consumiendo `Sequence[InfoTrial]`; no conoce JSONL.

Layout:

```
{raiz}/{run_id}/manifiesto.json   # metadatos PRED
{raiz}/{run_id}/backend.jsonl     # trials Optuna finalizados
```

## Rationale

- Un solo registro de trials (Optuna JSONL) evita divergencia entre
  manifiesto y backend.
- El controlador se prueba con `FakePuerto` in-memory sin levantar Optuna.
- Cambiar de backend HPO implica un nuevo adaptador, no reescribir el
  ciclo de vida de corrida.
- `ejecutar_estudio` integra el controlador en el punto de entrada HPO,
  no en `classical_selection.py` (cableado posterior).

## Consequences

- AST / pruebas verifican que `control_reanudacion` no importa `optuna` ni
  `adaptador_optuna`.
- `volcar_jsonl` filtra trials `RUNNING`; el generico asume que el trial
  activo se reejecuta en la frontera de trial.
- `COMPLETADA` + huella compatible reconstruye `ResultadoEstudio` sin
  reabrir el estudio.
- `FALLIDA` es terminal; `IncompatibilidadCorridaError` no muta el
  manifiesto.
- La sincronizacion del `RandomSampler` tras `add_trial` vive en el
  adaptador Optuna (detalle de backend, no del contrato generico).

## Alternatives Considered

- **Option A (persistencia monolitica en el controlador):** rechazado;
  duplica trials y acopla el generico a formato Optuna.
- **Segundo JSON de trials en el manifiesto:** rechazado; dos fuentes de
  verdad.
- **Importar Optuna en `control_reanudacion` para simplificar:** rechazado;
  rompe el mandato de reutilizacion vertical del Modulo 2.
