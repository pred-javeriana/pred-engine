# ADR-008: Router + Strategy sin estado (Modulo 2)

**Date:** 2026-09-20
**Status:** Proposed
**Notion ADR:** ADR-02-003
**Notion Task:** TASK-SEL-2.1-A1 / A2 / B1

## Context

El Modulo 1 entrega series con `sku_class`
(`smooth`, `intermittent`, `erratic`, `lumpy`). El Modulo 2 debe decidir
que familias de predictores se configuran y delegar a la estrategia
adecuada. Clasicos, ML y DL usan HPO; los fundacionales no.

Si el enrutador ejecuta Optuna, Walk-Forward o modelos, acopla "que
estrategia corre" con "como corre". Eso impide sustituir el backend HPO y
probar el routing con dobles.

`sku_class` no elige el modelo ganador; solo reduce candidatos.

## Decision

Se adopta **Router + Strategy sin estado**.

`SelectionRouter` solo:

1. recibe `SelectionRequest` (incluye `sku_class`);
2. consulta `RoutingPolicy`;
3. resuelve estrategias en `StrategyRegistry`;
4. delega `select(request, profile)`;
5. devuelve `tuple[SelectionResult, ...]`.

No lee Parquet, no importa Optuna, no implementa Walk-Forward, ASHA ni
entrenamiento, no elige el ganador del benchmark.

Estrategias y politica se inyectan por `Protocol` estructural (mismo
patron que `TrialHPO` en el motor HPO).

Contratos de frontera: Pydantic frozen, reutilizando `SkuClass` y
`ClassifiedObservation` del Modulo 1. Familias de contrato:
`classical | ml | dl | foundation`.

## Rationale

- El stub `optimizacion/router/` ya existia; no se crea un paquete paralelo.
- Title Case Notion (`Smooth`) romperia el Parquet 1.4 (ADR-003 / ADR-006).
- Un registro indexado por familia evita `if family == ...` en el router.

## Consequences

- El routing se prueba con `FakeStrategy` sin levantar HPO.
- Cambiar TPE/ASHA no toca el router.
- Registro y politica mal configurados fallan explicito
  (`UnregisteredFamilyError`, `UnknownSkuClassError`, overwrite sin
  `replace=True`).
- Hay que versionar la politica (ADR-009).

## Alternatives Considered

- **Router monolitico con HPO interno:** rechazado (acopla routing y
  optimizacion; ADR Notion Option A).
- **ABC de herencia para estrategias:** rechazado; los dobles de prueba
  y el HPO vigente usan typing estructural.
- **Familias `clasicos` / `CLASSICAL`:** rechazado. `clasicos` es jerga
  del adaptador HPO; `CLASSICAL` no coincide con el estilo `sku_class`.
