# ADR-009: Politica declarativa de routing por `sku_class`

**Date:** 2026-09-20
**Status:** Accepted
**Notion ADR:** ADR-02-012
**Notion Task:** TASK-SEL-2.2-A1 / A2 / B1

## Context

La seccion 2.2 usa `sku_class` para reducir familias candidatas. Eso no
es una regla "Smooth → ARIMA". Si la matriz vive como `if/elif` dentro
de `SelectionRouter`, cada cambio experimental obliga a tocar
infraestructura y borra la identidad de la politica que produjo una
corrida.

Hay que separar **mecanismo** (router) y **contenido** (matriz
versionada).

## Decision

Politica **declarativa e inyectable**.

- Contrato `RoutingPolicy`: `version` + `decide(sku_class) -> tuple[RoutingDecision, ...]`.
- `RoutingDecision` = `family` + `profile`.
- Implementacion inicial `InitialTopologyPolicy`, version
  `2.2.0-initial`, mapas `PROFILE_BY_SKU_CLASS` y
  `FAMILIES_BY_SKU_CLASS`.
- El router no contiene la matriz. Sella `policy_version` en cada
  `SelectionResult` y registra `(sku_id, clase, familia, perfil, politica)`.

Matriz inicial:

| `sku_class` | Perfil | Familias |
| --- | --- | --- |
| `smooth` | `dense_stable` | classical, ml, dl, foundation |
| `erratic` | `dense_variable` | classical, ml, dl, foundation |
| `intermittent` | `sparse_stable` | classical, ml, foundation |
| `lumpy` | `sparse_variable` | classical, foundation |

Clase desconocida, familia no registrada o politica vacia → error
explicito. Sin fallback. Literales de clase = contrato 1.4 minusculo.

## Rationale

Una matriz de datos es auditable y reemplazable (p. ej. tras el
benchmark) sin reescribir el router. La version evita que dos corridas
con matrices distintas se confundan.

## Consequences

- Pruebas parametrizadas de las cuatro clases.
- Una estrategia registrada pero excluida por la politica no se invoca
  (Lumpy no llama ML/DL).
- Cambiar la matriz exige bump de `INITIAL_POLICY_VERSION` (o una clase
  de politica nueva) para no reescribir el significado de corridas
  previas.
- La politica es un artefacto de configuracion que hay que versionar.

## Alternatives Considered

- **`if/elif` en el router:** rechazado (mezcla politica e
  infraestructura; Notion Option A).
- **Motor de reglas externo:** rechazado (complejidad sin necesidad
  experimental; Notion Option C).
- **Fallback a todas las familias si la clase es desconocida:**
  rechazado (fail-open; contradice 1.4 y ADR-005).
