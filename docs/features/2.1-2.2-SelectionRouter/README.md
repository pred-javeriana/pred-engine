# Seleccion 2.1 / 2.2 — SelectionRouter

## Que se hizo en esta sesion

Compilacion del **motor de enrutamiento** del Modulo 2 (tareas
`TASK-SEL-2.1-A1/A2/B1` y `TASK-SEL-2.2-A1/A2/B1`). El codigo listo para
pegar vive en `project/current-plan/IMPLEMENTATION.md`. Este directorio
documenta el diseno ya anclado al repo real (contrato 1.4 + stub
`optimizacion/router/`).

1. **Contratos Pydantic frozen:** `SelectionRequest`, `SelectionResult`,
   `RoutingDecision`. Familias `classical|ml|dl|foundation`. Perfiles
   `dense_stable|dense_variable|sparse_stable|sparse_variable`.
2. **`SelectionStrategy` y `RoutingPolicy` como `Protocol`.** Igual que
   `TrialHPO` en HPO: estructurales, sin ABC ni import de Optuna.
3. **`StrategyRegistry`:** indice familia → estrategia. Rechaza ausentes,
   desalineacion `family`/`clave`, y overwrite sin `replace=True`.
4. **`InitialTopologyPolicy`:** matriz declarativa versionada
   (`2.2.0-initial`) sobre las cuatro `sku_class` del Modulo 1.
5. **`SelectionRouter`:** sin estado, sin I/O, sin HPO. Consulta politica,
   resuelve registro, delega, sella `policy_version`, registra cada
   decision.

## Como

```
Parquet 1.4 (processed/)     ← frontera de I/O, NO es el router
        │
        ▼  (adaptador futuro / tests)
SelectionRequest(sku_id, sku_class, series?)
        │
        ▼
SelectionRouter
  ├─ RoutingPolicy.decide(sku_class) → (RoutingDecision…)
  ├─ StrategyRegistry.resolve(family)
  └─ strategy.select(request, profile) → SelectionResult
        │
        ▼
tuple[SelectionResult, …]  →  estrategias HPO / foundation (sesiones 2.3+)
```

`sku_class` **no se recalcula**. Se consume el literal persistido por 1.4
(`smooth|intermittent|erratic|lumpy`). ADI/CV² no cruzan esta frontera.

| `sku_class`    | Perfil            | Familias |
| -------------- | ----------------- | -------- |
| `smooth`       | `dense_stable`    | classical, ml, dl, foundation |
| `erratic`      | `dense_variable`  | classical, ml, dl, foundation |
| `intermittent` | `sparse_stable`   | classical, ml, foundation |
| `lumpy`        | `sparse_variable` | classical, foundation |

## Donde

| Pieza | Ruta |
| --- | --- |
| Plan compilado | `project/current-plan/IMPLEMENTATION.md` |
| Paquete | `src/pred_engine/optimizacion/router/` |
| Contrato 1.4 reutilizado | `src/pred_engine/comun/modelos/contrato.py` (`SkuClass`, `ClassifiedObservation`) |
| Pruebas | `tests/optimizacion/router/` |
| API | `API_SPECIFICATION.md` (este directorio) |
| ADR GitHub | `docs/adr/ADR-008-router-strategy.md`, `docs/adr/ADR-009-politica-declarativa.md` |
| Notion | ADR-02-003, ADR-02-012 |

## Relacion con Ingesta (1.3 / 1.4)

- 1.3 etiqueta cada SKU una sola vez (`classify_panel`).
- 1.4 publica exactamente cinco columnas en `{PRED_DATA_ROOT}/processed/*.parquet`.
- 2.1/2.2 leen **la etiqueta**, no el motor SBC. `read_classified_parquet`
  sigue en `ingesta.salida`; el router recibe `SelectionRequest` ya tipado.
- `series` opcional reutiliza `ClassifiedObservation` para no inventar un
  segundo esquema de filas.

## Mejoras respecto al texto de Notion (y por que)

- **`sku_class` minusculo.** Notion escribe `Smooth`. El Parquet y
  ADR-003/006 ya fijaron `smooth`. Title Case romperia el handoff.
- **Familias `classical|ml|dl|foundation`.** No `CLASSICAL` ni `clasicos`.
  El segundo es jerga interna de HPO (`classical_selection.py`).
- **`select(request, profile)` en vez de `select(request, decision)`.**
  La estrategia no necesita la politica; el router traduce
  `RoutingDecision → profile`.
- **2.2-A1 se compila antes de 2.1-B1.** El router consulta
  `RoutingPolicy`; un stub inventado violaria el dual-vision mandate.
- **Registro fail-closed.** Sin `replace=True` no se pisa una estrategia.
  Evita el fallo silencioso que ADR-02-003 senala como riesgo.
- **`policy_version` sellada por el router.** ADR-02-012 pide identificar
  la politica de la corrida; las estrategias no deben conocerla.
- **Coverage omit estrecha.** `optimizacion/*` entero estaba fuera del
  umbral 80 %. El router se mide; HPO y reanudacion siguen omitidos.
- **`git add` por rutas.** Evita mezclar cambios ajenos en el arbol.

## Fuera de alcance

- HPO / TPE / ASHA (`optimizacion/optimizadores/HPO/`)
- Walk-Forward (`comun/walkforward/`)
- Seleccion clasica SARIMA (`classical_selection.py`)
- Lectura de Parquet (sigue en 1.4)
- Control de reanudacion 2.9 (`boundary_constraints.md`)
- Estrategias reales de ML / DL / foundation
