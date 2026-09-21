# Seleccion 2.4 — HPO (TPE Sampler + ASHA Resource Allocator)

## Que se hizo en esta sesion

Reconciliacion de las tareas Notion del Modulo 2.4 (`TASK-HPO-4.0-*` y
`TASK-HPO-5.0-*`, asignadas a Tomas Ramirez Roa) contra la implementacion
real del motor de HPO, que ya existia -- construida en sesiones anteriores
bajo otra numeracion de tareas (`TASK-SEL-2.81-*`, `TASK-SEL-2.9-*`, ya
mergeadas a `main`). El motor vive en
`src/pred_engine/optimizacion/optimizadores/HPO/` desde antes de esta
sesion; este documento y ADR-012/ADR-013 formalizan retroactivamente las
decisiones de diseño que Notion aun no reflejaba, y esta sesion cierra las
brechas funcionales que si estaban genuinamente ausentes:

1. Tipo `Ordinal` en `espacio.py` (codificacion explicita de ordinales como
   enteros ordenados).
2. Warm start (`adaptador_optuna.py::semillas_desde_historico` /
   `inyectar_historico`, expuesto como `EstudioHPO.agregar_trials_historicos`)
   para inyectar evidencia de una corrida externa (SKUs lumpy).
3. `timestamp` por trial (`Trial.timestamp`), completando el formato de
   historial `{config, metric_value, timestamp, status}`.
4. Benchmark TPE vs random search sobre sphere/Rastrigin/Rosenbrock
   (`benchmarks/`).
5. Cuantificacion del ahorro de ASHA (>= 30 %) y verificacion de que toda
   poda queda auditada con `motivo`.
6. Advertencia explicita (antes silenciosa) cuando se reanuda con
   `TPESampler`: su RNG de arranque en frio no se resincroniza (limitacion
   conocida de Optuna, ver ADR-012).
7. `raiz_corrida`/`run_id` cableados en `classical_selection.py` (pendiente
   documentado en `GUIA_CICLO_VIDA_CORRIDA.md`).

**Fuera de alcance de esta version** (decision de arquitectura, no deuda
pendiente -- ver ADR-013): un registro compartido thread-safe y un
asignador de *workers* concurrentes (`ThreadPoolExecutor`, locks) para
paralelizar trials de UN estudio. La version actual ejecuta un bucle
secuencial de un solo proceso; el paralelismo real del repo ocurre a nivel
de SKU (`classical_selection.py::seleccionar_por_panel`,
`ProcessPoolExecutor`).

## Como

```
EspacioBusqueda (numerico/categorico/ordinal/condicional)
        │
        ▼
construir_muestreador_tpe()  ──►  optuna.samplers.TPESampler
        │
        ▼
ejecutar_estudio()  (bucle secuencial ask/tell)
        │
        ├─ EjecutorGreedy (Walk-Forward, ventana a ventana)
        │      │
        │      ▼
        │  es_degenerada()  ── poda semantica
        │
        ├─ DecisorASHA.decidir()  ── poda por escalon (>= min_ventanas)
        │
        ├─ agregar_trials_historicos()  ── warm start (opcional)
        │
        └─ ControladorReanudacion (opcional, ver 2.9)
```

| Pieza | Responsabilidad | Archivo |
| --- | --- | --- |
| `EspacioBusqueda` / `Entero` / `Flotante` / `Categorico` / `Ordinal` / `Condicion` | DSL del espacio de busqueda | `espacio.py` |
| `construir_muestreador_tpe` | TPE (delegado a Optuna, ADR-012) | `muestreadores.py` |
| `DecisorASHA` / `ReglasPoda` / `es_degenerada` | Decision de poda ASHA, pura (ADR-013) | `asha.py`, `poda.py` |
| `contratos.py` | Frontera `Protocol` sin `import optuna` | `contratos.py` |
| `adaptador_optuna.py` | Unico modulo que importa Optuna: persistencia, warm start, poda concreta | `adaptador_optuna.py` |
| `registro.py` | Traduce `InfoTrial` → `Trial`/`ResultadoEstudio` | `registro.py` |
| `estudio.py` | Punto de entrada unico | `estudio.py` |
| `benchmarks/` | Sphere/Rastrigin/Rosenbrock, comparacion TPE vs random | `benchmarks/` |

## Resultados de benchmark (TASK-HPO-4.0-C1 / 5.0-C1)

Reproducible con `uv run pytest tests/optimizacion/HPO/benchmarks -q -m slow`.

### TPE vs random search (dim=3, 60 trials, seed=0)

| Funcion | Mejor valor TPE | Mejor valor random | TPE converge mejor |
| --- | --- | --- | --- |
| sphere | 0.230 | 2.044 | si |
| rastrigin | 13.94 | 21.46 | si |
| rosenbrock | 2.09 | 11.05 | si |

Resultado estable en semillas adicionales (0-3): TPE supera a random search
en las 3 funciones en todos los casos probados.

### ASHA vs sin poda (n_trials=30, `ReglasPoda(min_ventanas=4, factor_reduccion=2)`)

Proxy de costo: ventanas Walk-Forward evaluadas (proporcional al tiempo
real; evita la varianza de medir reloj en CI compartido).

| Modo | Ventanas evaluadas | Reduccion |
| --- | --- | --- |
| Sin poda | 1200 | — |
| Con ASHA | 620 | 48.3 % (>= 30 % exigido) |

Todas las podas quedan registradas con `motivo` no vacio
(`test_todos_los_trials_podados_tienen_motivo_auditable`).

## Verificacion

```bash
uv run pytest tests/optimizacion/HPO -q
uv run pytest tests/optimizacion/HPO/benchmarks -q -m slow
uv run pytest tests/optimizacion/modelos_clasicos -q
uv run ruff check src/pred_engine/optimizacion/optimizadores/HPO tests/optimizacion/HPO
uv run pyright src/pred_engine/optimizacion/optimizadores/HPO
```

## Pendiente conocido (heredado, no cerrado por esta sesion)

Ver `GUIA_CICLO_VIDA_CORRIDA.md`: el stub `forecasting/control_reanudacion/`
es de otro modulo y no se toca aqui.
