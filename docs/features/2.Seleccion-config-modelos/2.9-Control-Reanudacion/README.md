# Seleccion 2.9 — Control de reanudacion y reproducibilidad

## Que se hizo en esta sesion

Implementacion del **control de reanudacion** del Modulo 2 (tareas
`TASK-SEL-2.9-A1/A2/B1` y `TASK-SEL-2.9-C1/C2`). El codigo generico vive
en `src/pred_engine/optimizacion/control_reanudacion/`; la persistencia
Optuna en `src/pred_engine/optimizacion/optimizadores/HPO/adaptador_optuna.py`.
El plan TDD compilado esta en `project/current-plan/IMPLEMENTATION.md`.

1. **Manifiesto versionado y maquina de estados:** `ManifiestoCorrida`,
   `EstadoCorrida`, transiciones fail-closed con identidad en
   `EN_PROGRESO` y `COMPLETADA`.
2. **Huella determinista:** SHA-256 sobre configuracion experimental
   (incluye `n_observaciones`, `huella_serie`, espacio canonico y reglas
   de poda). `run_id` no entra en la huella.
3. **Almacenamiento atomico:** `AlmacenManifiestosFs` escribe
   `manifiesto.json` via `*.tmp` + `os.replace`.
4. **Puerto del motor HPO:** `PuertoPersistenciaMotor` implementado por
   `AdaptadorPersistenciaOptuna` (reusa `volcar_jsonl` / `reanudar_estudio`).
5. **Controlador:** `ControladorReanudacion` orquesta crear, reanudar,
   rechazar, checkpoint, interrumpir, completar y fallar.
6. **Integracion HPO:** `ejecutar_estudio` acepta `raiz_corrida` /
   `run_id` / `controlador` opcionales; persiste en frontera de trial.

## Como

```
ejecutar_estudio(..., raiz_corrida, run_id)
        │
        ▼
ControladorReanudacion.abrir(SolicitudCorrida)
  ├─ AlmacenManifiestosFs  →  {raiz}/{run_id}/manifiesto.json
  ├─ verificar_compatibilidad (huella)
  └─ PuertoPersistenciaMotor  →  {raiz}/{run_id}/backend.jsonl
        │
        ▼
AdaptadorPersistenciaOptuna (unico modulo que importa optuna)
  ├─ crear / restaurar Study
  ├─ volcar_jsonl (solo COMPLETE / PRUNED / FAIL)
  └─ reanudar_estudio (+ sincronizar RandomSampler)
        │
        ▼
ResultadoEstudio  (via registro.instantanea_desde_estudio)
```

| Pieza | Responsabilidad |
| --- | --- |
| `control_reanudacion/` | Ciclo de vida PRED, manifiesto, huella. **Sin** `import optuna`. |
| `adaptador_optuna.py` | Serializacion JSONL, restauracion del `Study`, pruner ASHA. |
| `estudio.py` | Bucle ask/tell, checkpoint tras cada trial, manejo de interrupcion. |

### Estados y transiciones

| Estado | Significado | Reanudable |
| --- | --- | --- |
| `nueva` | Manifiesto recien creado | → `en_progreso` |
| `en_progreso` | Corrida activa o recuperacion no senalizada (kill -9) | si |
| `interrumpida` | Interrupcion senalizada (`KeyboardInterrupt`, etc.) | si |
| `completada` | Objetivo de trials alcanzado | reconstruye resultado, no reejecuta |
| `fallida` | Error no recuperable durante la corrida | no |

Transiciones permitidas:

| Desde | Hacia |
| --- | --- |
| `nueva` | `en_progreso` |
| `en_progreso` | `en_progreso` (identidad / checkpoint), `completada`, `fallida`, `interrumpida` |
| `interrumpida` | `en_progreso`, `fallida` |
| `completada` | `completada` (identidad) |
| `fallida` | (terminal) |

### Granularidad de recuperacion

- **Frontera de trial:** el checkpoint se escribe despues de cada trial
  finalizado (`tell` completado).
- **Trial RUNNING:** si la interrupcion ocurre a mitad de walk-forward, el
  trial activo **no** se serializa en `backend.jsonl`. Al reanudar se
  reejecuta desde cero con el mismo numero de trial Optuna.
- **No** hay reanudacion intra-ventana (ADR-011).

## Donde

| Pieza | Ruta |
| --- | --- |
| Plan compilado | `project/current-plan/IMPLEMENTATION.md` |
| Paquete generico | `src/pred_engine/optimizacion/control_reanudacion/` |
| Adaptador Optuna | `src/pred_engine/optimizacion/optimizadores/HPO/adaptador_optuna.py` |
| Punto de entrada HPO | `src/pred_engine/optimizacion/optimizadores/HPO/estudio.py` |
| Pruebas genericas | `tests/optimizacion/control_reanudacion/` |
| Prueba integral | `tests/optimizacion/HPO/test_estudio_reanudacion.py` |
| API | `API_SPECIFICATION.md` (este directorio) |
| ADR GitHub | `docs/adr/ADR-010-control-corrida-vs-backend.md`, `docs/adr/ADR-011-checkpoints-atomicos-idempotentes.md` |
| Notion | ADR-02-013, ADR-02-014 |

## Mejoras respecto al texto de Notion (y por que)

- **Reutilizar `volcar_jsonl` / `reanudar_estudio`.** Un segundo registro
  de trials romperia ADR-02-013 Option B. El manifiesto solo guarda la
  referencia `backend.jsonl`.
- **`volcar_jsonl` omite RUNNING.** Optuna rechaza `add_trial` con RUNNING;
  el trial incompleto se reejecuta en la frontera de trial.
- **Escritura atomica tambien en JSONL.** `escribir_atomico` protege
  manifiesto y backend ante interrupcion durante el volcado.
- **`EN_PROGRESO` reanudable.** Un kill -9 no escribe `interrumpida`; sin
  esto las caidas duras serian irrecuperables.
- **Incompatibilidad no muta el manifiesto.** El rechazo por huella deja
  el artefacto intacto para auditoria.
- **Transicion identidad legal.** Checkpoints repetidos sobre `en_progreso`
  no violan la maquina de estados (idempotencia ADR-02-014).
- **Huella incluye serie y longitud.** `huella_serie` (SHA-256 float64) y
  `n_observaciones` evitan reanudar sobre otra `y`.
- **`run_id` fuera de la huella.** Identifica la corrida, no el experimento.
- **`EspacioBusqueda.descripcion_canonica()`.** Las restricciones callable
  no son JSON; se serializan por `qualname` o hash de bytecode.
- **`ejecutar_estudio` con kwargs opcionales.** Sin `raiz_corrida` el
  comportamiento previo no cambia (tests HPO existentes intactos).
- **`COMPLETADA` reconstruye sin reabrir.** Huella compatible →
  `instantanea_desde_estudio` sin nuevos trials.
- **Identificadores en espanol** (`ControladorReanudacion`, `ejecutar_estudio`).
- **`_sincronizar_sampler_aleatorio` en reanudacion.** Tras `add_trial`,
  un `RandomSampler` nuevo repite la primera muestra; se avanza el RNG
  por `n_trials_cargados` antes del primer `ask()` post-restore.
- **Coverage omit estrecho.** `control_reanudacion/` se mide; `optimizadores/*`
  sigue omitido salvo pruebas de integracion.
- **`git add` por rutas.** Evita mezclar cambios ajenos.

## Verificacion

```bash
uv run pytest tests/optimizacion/control_reanudacion \
  tests/optimizacion/HPO/test_estudio.py \
  tests/optimizacion/HPO/test_estudio_reanudacion.py \
  tests/optimizacion/HPO/test_adaptador_optuna.py -q

uv run pytest --cov=pred_engine --cov-fail-under=80 -q

uv run ruff check src/pred_engine/optimizacion/control_reanudacion \
  src/pred_engine/optimizacion/optimizadores/HPO/estudio.py \
  src/pred_engine/optimizacion/optimizadores/HPO/adaptador_optuna.py
```

- `tests/optimizacion/control_reanudacion/`: 29 pruebas.
- `tests/optimizacion/HPO/test_estudio_reanudacion.py`: 4 pruebas (incluye
  interrupcion simulada vs corrida continua).
- Comando del paso 2.9: **66 pruebas** pasan.
- Suite completa: **437 pruebas**, cobertura global **91,6 %**.
- `pyproject.toml`: omit de `control_reanudacion` retirado; pyright incluye
  el paquete; `pythonpath = ["."]` para imports de `tests._dobles`.

## Fuera de alcance

- Reanudacion intra-ventana walk-forward
- `classical_selection.py` (sigue llamando `ejecutar_estudio` sin
  `raiz_corrida` hasta que una sesion posterior lo cablee)
- Stub `forecasting/control_reanudacion/` (otro modulo; no tocar)
- Router 2.1/2.2, politica topologica, ASHA puro, `EjecutorGreedy`
- Lectura de Parquet 1.4
- Persistencia de estado del muestreador TPE (solo `RandomSampler` sincronizado)
