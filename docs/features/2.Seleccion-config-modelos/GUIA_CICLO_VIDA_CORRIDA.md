# Guia — Ciclo de vida de corrida (Modulo 2)

Playbook para quien extienda el Modulo 2 (estrategias, `classical_selection`,
nuevos motores) sin romper reanudacion ni reproducibilidad.

**Lectura complementaria (no duplicar aqui):**

| Documento | Para que |
| --- | --- |
| `2.9-Control-Reanudacion/README.md` | Que se implemento y donde |
| `2.9-Control-Reanudacion/API_SPECIFICATION.md` | Contratos y firmas |
| `2.4-HPO/README.md` | TPE/ASHA: que se implemento, benchmarks |
| `2.4-HPO/API_SPECIFICATION.md` | Contratos y firmas de HPO (warm start, `Ordinal`) |
| `docs/adr/ADR-010-*.md` | Manifiesto vs backend Optuna |
| `docs/adr/ADR-011-*.md` | Huella, atomico, frontera de trial |
| `docs/adr/ADR-012-*.md` | TPE delegado a Optuna, no reimplementado |
| `docs/adr/ADR-013-*.md` | ASHA secuencial, sin registro multi-worker |

---

## Reglas invariantes

Antes de mergear cualquier cambio en Modulo 2 que ejecute HPO, verifica:

1. **`control_reanudacion/` no importa Optuna.** Ni `optuna`, ni
   `adaptador_optuna`. El generico solo conoce `PuertoPersistenciaMotor`
   (handle opaco).

2. **Un solo registro de trials.** Los trials viven en
   `backend.jsonl` (Optuna). El manifiesto guarda **solo la referencia**
   (`backend_checkpoint`), nunca una copia paralela de trials.

3. **Checkpoint en frontera de trial.** Se persiste despues de cada
   `tell` exitoso (completado / podado / fallido). **No** hay
   reanudacion intra-ventana walk-forward.

4. **Trial RUNNING no se serializa.** Si la corrida cae a mitad de un
   trial, ese trial se reejecuta entero al reanudar. Es intencional
   (ADR-011).

5. **`run_id` identifica la corrida; la huella identifica el experimento.**
   Misma config + distinto `run_id` = dos corridas validas. Mismo
   `run_id` + config distinta = rechazo (`IncompatibilidadCorridaError`)
   **sin** mutar el manifiesto.

6. **Escritura atomica en disco.** Manifiesto y JSONL usan `*.tmp` +
   `os.replace`. No escribas directo al archivo final.

7. **Estados reanudables:** `en_progreso` (incluye kill -9 no senalizado)
   e `interrumpida`. `completada` reconstruye resultado sin reejecutar.
   `fallida` es terminal.

---

## Arbol de responsabilidades

```
Caller (estrategia, classical_selection, CLI futuro)
    │
    ▼  raiz_corrida + run_id  (o controlador inyectado)
ejecutar_estudio()
    │
    ├─ ControladorReanudacion     ← generico, sin Optuna
    │     ├─ AlmacenManifiestosFs → manifiesto.json
    │     └─ PuertoPersistenciaMotor
    │
    └─ AdaptadorPersistenciaOptuna ← unico import de optuna
          └─ backend.jsonl
```

El **caller** decide *donde* persistir (`raiz_corrida`) y *que corrida*
es (`run_id`). El **controlador** decide *crear / reanudar / rechazar*.
El **adaptador** sabe *como* serializar Optuna.

---

## Receta: cablear un nuevo punto de entrada

Cuando una estrategia o `classical_selection` llame a HPO con
persistencia:

### 1. Elegir `run_id` estable

Convencion sugerida (ajustar al producto, pero ser consistente):

```
{familia}-{sku_id}-{sesion_o_timestamp}
```

El `run_id` debe ser el mismo entre interrupcion y reanudacion.

### 2. Pasar kwargs a `ejecutar_estudio`

```python
resultado = ejecutar_estudio(
    y,
    espacio,
    fabrica,
    n_trials=...,
    min_train=...,
    horizonte=...,
    paso=...,
    metrica_objetivo=...,
    estacionalidad=...,
    seed=...,
    familia=...,
    sku_id=...,
    muestreador=...,
    reglas=...,
    raiz_corrida="/ruta/a/corridas",  # nuevo
    run_id="classical-sku-42-20260920",  # nuevo
)
```

Sin `raiz_corrida` el comportamiento es el de siempre: **no escribe
disco**. Util para tests rapidos.

### 3. No construir `SolicitudCorrida` en el caller (salvo tests)

`ejecutar_estudio` ya arma la solicitud internamente a partir de los
mismos argumentos. Solo inyecta `controlador` manualmente si necesitas
un doble en pruebas.

### 4. Layout esperado en disco

```
{raiz_corrida}/
  {run_id}/
    manifiesto.json      # metadatos PRED + estado + huella
    backend.jsonl        # trials Optuna finalizados
```

No muevas ni renombres estos archivos a mano durante una corrida activa.

### 5. Reanudar = misma llamada

Para continuar tras interrupcion, invoca `ejecutar_estudio` con los
**mismos** argumentos experimentales y el **mismo** `run_id`. El
controlador detecta el manifiesto y restaura el estudio.

Si cambias `seed`, `y`, espacio, reglas de poda o metrica bajo el mismo
`run_id`, la huella rechazara la reanudacion.

---

## Receta: anadir una estrategia de seleccion (2.3+)

1. La estrategia implementa `SelectionStrategy.select()` (router 2.1).
2. Dentro de `select`, si usa HPO, llama a `ejecutar_estudio` — no
   dupliques el bucle ask/tell.
3. Pasa `raiz_corrida` / `run_id` cuando la corrida deba ser
   recuperable (produccion, benchmarks largos).
4. **No** importes `control_reanudacion` desde el router; solo desde la
   estrategia o el adaptador de ejecucion si hace falta inyectar un
   controlador de prueba.

---

## Anti-patrones (no hacer)

| Anti-patron | Por que falla |
| --- | --- |
| `import optuna` en `control_reanudacion/` | Rompe ADR-010; acopla el generico al vendor |
| Segundo JSON de trials en el manifiesto | Dos fuentes de verdad |
| Persistir trial RUNNING en JSONL | Optuna rechaza `add_trial`; estado inconsistente |
| Mutar manifiesto al rechazar por huella | Viola AC de C1; pierde auditoria |
| Checkpoint dentro de `_correr_trial` / por ventana | Complejidad enorme; fuera de alcance 2.9 |
| Asumir 1 llamada a `fabrica()` = 1 trial | `fabrica` se invoca **por ventana** walk-forward |
| Reusar `run_id` con otra serie `y` | La huella de serie lo rechaza (bien) o, si no se cablea bien, resultados invalidos |
| `git add .` en commits de Modulo 2 | Mezcla cambios ajenos; usar rutas explicitas |

---

## Pruebas minimas al integrar

Copia el patron de `tests/optimizacion/HPO/test_estudio_reanudacion.py`:

| Caso | Que verifica |
| --- | --- |
| Sin `raiz_corrida` | No crea archivos en disco |
| Corrida nueva | Manifiesto `completada`, trials >= objetivo |
| Segunda llamada completada | No repite trials (ids y valores iguales) |
| Interrupcion + resume | Mismo conjunto de trials que corrida continua |
| Huella incompatible | `IncompatibilidadCorridaError`, manifiesto intacto |

Comando rapido del modulo 2.9:

```bash
uv run pytest tests/optimizacion/control_reanudacion \
  tests/optimizacion/HPO/test_estudio_reanudacion.py \
  tests/optimizacion/HPO/test_adaptador_optuna.py -q
```

AST de desacoplamiento (el generico no importa Optuna):

```bash
uv run pytest tests/optimizacion/control_reanudacion -q -k import
```

(Si no existe test AST dedicado, revisar manualmente que
`control_reanudacion/*.py` no mencione `optuna`.)

---

## Pendiente conocido (no olvidar al cablear)

| Item | Estado |
| --- | --- |
| `classical_selection.py` → `raiz_corrida` / `run_id` | **Cableado** (Modulo 2.4, ver `2.4-HPO/README.md`) |
| Sincronizacion de muestreador TPE al reanudar | Solo `RandomSampler` cubierto; con `TPESampler` ahora se advierte explicitamente (antes silencioso) — limitacion conocida, ver ADR-012 |
| Stub `forecasting/control_reanudacion/` | Otro modulo; **no tocar** |

`seleccionar_configuracion_clasica` ya acepta `raiz_corrida`/`run_id` (los
reenvia a `ejecutar_estudio`). `seleccionar_por_panel` **rechaza** `run_id`
explicito (colisionaria entre SKUs); para persistencia en modo panel, llamar
`seleccionar_configuracion_clasica` por SKU con un `run_id` propio.

---

## Checklist pre-PR (Modulo 2 + ciclo de vida)

- [ ] Sin `import optuna` fuera de `adaptador_optuna.py` (en el camino nuevo)
- [ ] Checkpoints solo tras trial finalizado
- [ ] `run_id` documentado o derivado de forma estable
- [ ] Prueba de no-escritura sin `raiz_corrida` (o justificacion si no aplica)
- [ ] Prueba de resume o nota en PR si el flujo nuevo aun no es reanudable
- [ ] `ruff check` + `pytest` en rutas tocadas
- [ ] README/API del sub-feature actualizado si cambian contratos publicos
