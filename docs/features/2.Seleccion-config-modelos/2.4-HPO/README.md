# Seleccion 2.4 — HPO (TPE Sampler + ASHA Resource Allocator)

## Capacidades

El motor de HPO vive en `src/pred_engine/optimizacion/optimizadores/HPO/` y
es comun a las familias de modelos. Sus decisiones de diseño estan en
ADR-012 y ADR-013.

1. `Ordinal` en `espacio.py` codifica ordinales como enteros ordenados.
2. El warm start (`adaptador_optuna.py::semillas_desde_historico` /
   `inyectar_historico`, expuesto como `EstudioHPO.agregar_trials_historicos`)
   importa evidencia de una corrida externa (SKUs lumpy).
3. Cada trial registra su `timestamp` (`Trial.timestamp`), con el formato de
   historial `{config, metric_value, timestamp, status}`.
4. `benchmarks/` compara TPE contra random search sobre
   sphere/Rastrigin/Rosenbrock.
5. ASHA ahorra al menos 30 % de ventanas evaluadas, y toda poda queda
   auditada con `motivo`.
6. Al reanudar con `TPESampler` el motor emite una advertencia, porque el RNG
   de arranque en frio no se resincroniza (limitacion de Optuna, ver
   ADR-012).
7. `classical_selection.py` recibe `raiz_corrida` y `run_id` para la
   reanudacion (ver `GUIA_CICLO_VIDA_CORRIDA.md`).

**Fuera de alcance** (decision de arquitectura -- ver ADR-013): un registro compartido thread-safe y un
asignador de *workers* concurrentes (`ThreadPoolExecutor`, locks) para
paralelizar trials de UN estudio. El motor ejecuta un bucle
secuencial de un solo proceso; el paralelismo real del repo ocurre a nivel
de SKU (`classical_selection.py::seleccionar_por_panel`,
`ProcessPoolExecutor`).

## Como

```
EspacioBusqueda (numerico/categorico/ordinal/condicional)
        │
        ▼
construir_muestreador_tpe(seed=derivar_semilla(seed, "familia/sku", 0))
        │                                   ──►  optuna.samplers.TPESampler
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
| `adaptador_optuna.py` | Adaptacion de Optuna: persistencia, warm start y poda concreta | `adaptador_optuna.py` |
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

## Recuperacion

La recuperacion de estudios HPO pertenece a `optimizacion/control_reanudacion`;
la coordinacion y recuperacion de M3 pertenece a
`forecasting/control_reanudacion`.
