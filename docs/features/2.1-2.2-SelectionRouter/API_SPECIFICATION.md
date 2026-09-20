# API — Seleccion 2.1 / 2.2 (SelectionRouter)

Identificadores en ingles; comentarios y mensajes en espanol. El paquete
no importa Optuna, Walk-Forward ni modelos concretos.

Punto de entrada: `pred_engine.optimizacion.router`.

## Relacion con el artefacto 1.4

El router **no** abre `{PRED_DATA_ROOT}/processed/*.parquet`. Consume:

| Campo | Origen 1.4 | Notas |
| --- | --- | --- |
| `sku_id` | columna String | no vacio, strip |
| `sku_class` | columna String | `smooth` \| `intermittent` \| `erratic` \| `lumpy` |
| `series` | filas `ClassifiedObservation` | opcional; si viene, identidad coherente |

No se aceptan sinonimos Notion (`Smooth`, `intermitente`, `Irregular`).

## Literales

### `PredictorFamily`

`Literal["classical", "ml", "dl", "foundation"]` — tupla
`PREDICTOR_FAMILIES`.

### `TopologicalProfile`

`Literal["dense_stable", "dense_variable", "sparse_stable", "sparse_variable"]`
— tupla `TOPOLOGICAL_PROFILES`.

### `SkuClass`

Reexport conceptual desde `pred_engine.comun.modelos`. No se redefine.

## Contratos Pydantic (`contratos.py`)

Todos: `strict=True`, `extra=forbid`, `frozen=True`.

### `SelectionRequest`

| Campo | Tipo | Default |
| --- | --- | --- |
| `sku_id` | `str` (min 1, no solo espacios) | — |
| `sku_class` | `SkuClass` | — |
| `series` | `tuple[ClassifiedObservation, ...]` | `()` |

Valida que cada observacion de `series` tenga el mismo `sku_id` y
`sku_class` que la solicitud.

### `SelectionResult`

| Campo | Tipo | Default |
| --- | --- | --- |
| `sku_id` | `str` | — |
| `sku_class` | `SkuClass` | — |
| `family` | `PredictorFamily` | — |
| `profile` | `TopologicalProfile` | — |
| `produced_by` | `str` (min 1) | — |
| `policy_version` | `str` | `""` (el router la sella) |
| `payload` | `Mapping[str, Any]` | `{}` opaco; el router no lo interpreta |

### `RoutingDecision`

`family: PredictorFamily` + `profile: TopologicalProfile`. Igualdad por
valor. Inmutable.

### `SelectionStrategy` (`Protocol`, `runtime_checkable`)

```
family: PredictorFamily
select(request: SelectionRequest, profile: TopologicalProfile) -> SelectionResult
```

No hay clase base que heredar. Un doble de prueba con esos miembros basta.

### `RoutingPolicy` (`Protocol`, `runtime_checkable`)

```
version: str
decide(sku_class: SkuClass) -> tuple[RoutingDecision, ...]
```

## Registro (`registro.py`)

### `StrategyRegistry`

- `register(family, strategy, *, replace=False) -> None`
- `resolve(family) -> SelectionStrategy`
- `families -> frozenset[PredictorFamily]`

Errores:

| Condicion | Excepcion |
| --- | --- |
| familia fuera de `PREDICTOR_FAMILIES` | `SelectionContractError` |
| objeto sin contrato `SelectionStrategy` | `SelectionContractError` |
| `strategy.family != family` | `SelectionContractError` |
| clave ocupada y `replace=False` | `DuplicateStrategyError` |
| `resolve` de familia ausente | `UnregisteredFamilyError` |

`replace=True` emite `logger.warning`. Independiente de `RoutingPolicy`.

## Politica (`politica.py`)

### Constantes

- `INITIAL_POLICY_VERSION = "2.2.0-initial"`
- `PROFILE_BY_SKU_CLASS`
- `FAMILIES_BY_SKU_CLASS`

Al importar se valida que las cuatro `SKU_CLASSES` tengan perfil y al
menos una familia, sin duplicados.

### `InitialTopologyPolicy`

`version` de clase. `decide(sku_class)`:

| Clase | Perfil | Familias (orden estable) |
| --- | --- | --- |
| `smooth` | `dense_stable` | classical, ml, dl, foundation |
| `erratic` | `dense_variable` | classical, ml, dl, foundation |
| `intermittent` | `sparse_stable` | classical, ml, foundation |
| `lumpy` | `sparse_variable` | classical, foundation |

`sku_class` desconocida (incl. `Smooth` Title Case) →
`UnknownSkuClassError`. Sin fallback.

## Enrutador (`enrutador.py`)

### `SelectionRouter(policy, registry)`

Sin estado entre `route()`. Solo guarda `_policy` y `_registry`.

`route(request: SelectionRequest) -> tuple[SelectionResult, ...]`:

1. Rechaza si `request` no es `SelectionRequest`.
2. `decisiones = policy.decide(sku_class)`; vacio →
   `RouterConfigurationError`.
3. Por cada decision: `registry.resolve(family)` →
   `strategy.select(request, profile)`.
4. Verifica identidad (`sku_id`, `sku_class`, `family`, `profile`).
5. Sella `policy_version` si venia vacia.
6. `logger.info` por decision: `sku_id`, `clase`, `familia`, `perfil`,
   `politica`. No registra `demand_qty`.

No lee ni escribe Parquet. No importa `optimizacion.optimizadores`.

## Excepciones (`errores.py`)

Jerarquia, todas subclases de `ValueError`:

```
SelectionError
├─ SelectionContractError
├─ UnregisteredFamilyError
├─ DuplicateStrategyError
├─ UnknownSkuClassError
└─ RouterConfigurationError
```

## Logging

`pred_engine.comun.logger.get_logger(__name__)`. Eventos: registro /
reemplazo de estrategia, clase desconocida, enrutamiento por (SKU,
familia, perfil, version de politica). Sin PII ni series de demanda.
