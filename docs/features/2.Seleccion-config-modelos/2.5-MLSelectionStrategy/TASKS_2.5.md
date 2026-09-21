# Tareas 2.5 — MLSelectionStrategy

Fuente: Notion, pagina "6. MLSelectionStrategy" (hija de "Sprint 2: Modulo 2"),
8 tareas `TASK-MLS-6.0-*` asignadas a Tomas Ramirez Roa, `P1-Important`, estado
`Backlog` al momento de escribir esto. La pagina no traia tareas, subsecciones
ni ADRs: las tareas se derivaron de su especificacion (2.5) y del patron de
`classical_selection.py`. Todas pasan a `Completada` solo tras el PR aprobado.

**Hallazgo que enmarca todo:** la politica 2.2 enruta la familia `ml` solo a
`smooth`, `erratic` e `intermittent`; **`lumpy` queda fuera**. Las tareas
C1 y B3 se corrigieron en Notion en consecuencia.

---

## TASK-MLS-6.0-A1 — Disenar arquitectura y ADR de libreria base

**Estado: implementada.**

- [x] Contrato de la estrategia compatible con `SelectionStrategy` (router 2.1),
  sin importar Optuna (AST en `test_delegacion.py`).
- [x] El diseno no duplica el bucle ask/tell: todo pasa por `ejecutar_estudio`.
- [x] Libreria decidida: LightGBM. Features: lags + agregados + fase estacional.
  Pronostico recursivo. Presupuesto por perfil topologico.
- [x] ADR-014 redactado (estado `Proposed`).
- [ ] **Manual en Notion:** crear el ADR en "ADRs Pendientes" de la pagina 6,
  enlazar `Git .md URL` y pasarlo a `Aceptada` al merge.

**Archivos:** `docs/adr/ADR-014-ml-lightgbm-y-features-autoregresivos.md`.

---

## TASK-MLS-6.0-B1 — Espacio de busqueda de hiperparametros ML

**Estado: implementada.**

- [x] 9 hiperparametros (`lags`, `n_estimators`, `max_depth`, `learning_rate`,
  `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`).
- [x] Ninguna configuracion muestreada viola las restricciones.
- [x] `descripcion_canonica()` estable (participa en la huella de 2.9).

**Archivos:** `optimizadores/modelos_machine_learning/espacio_ml.py`,
`tests/optimizacion/modelos_machine_learning/test_espacio_ml.py`.

---

## TASK-MLS-6.0-B2 — Pronosticador ML y fabrica compatible con Walk-Forward

**Estado: implementada.**

- [x] `LightGBMForecaster` respeta `BaseForecaster` / `Pronosticador`.
- [x] Test de no fuga temporal (`construir_muestras`) y de que perturbar el
  futuro no altera el ajuste previo.
- [x] Mismo `seed` + configuracion = pronosticos identicos.
- [x] Series cortas -> `SerieCortaError`; demanda intermitente -> pronostico >= 0.
- [x] `lightgbm>=4.0` en `pyproject.toml`; ruta agregada a `[tool.pyright] include`.
- [~] Sin calendario real (la serie llega 1D): se usa fase `posicion mod m`.

**Archivos:** `comun/modelos/modelos_machine_learning/`,
`tests/comun/modelos/test_lgbm_forecaster.py`.

---

## TASK-MLS-6.0-B3 — `seleccionar_configuracion_ml` delegando en HPO

**Estado: implementada.**

- [x] Toda la optimizacion pasa por `ejecutar_estudio` (espia en
  `test_delegacion.py`).
- [x] La ganadora incluye hiperparametros, metrica, valor y `n_ventanas`.
- [x] Reanudacion con el mismo `run_id`; huella incompatible rechazada sin mutar
  el manifiesto.
- [x] Modo panel identico con 1 y N procesos.
- [x] Nucleo compartido `seleccion_hpo.py` extraido de `classical_selection.py`
  (clasicos migrado, 15 tests siguen verdes).
- [x] **Bug corregido:** el pool con `fork` se colgaba si el padre ya habia
  entrenado LightGBM (OpenMP). El pool usa ahora `spawn` (ver ADR-014, punto 9).
- [~] `historico_previo` (warm start) expuesto pero opcional: `lumpy` no llega a ML.

**Archivos:** `optimizadores/seleccion_hpo.py`,
`optimizadores/modelos_machine_learning/ml_selection.py`,
`comun/dataclasses/modelos_machine_learning.py`.

---

## TASK-MLS-6.0-B4 — `MLSelectionStrategy` registrada en el router

**Estado: implementada.**

- [x] `isinstance(MLSelectionStrategy(), SelectionStrategy)`; `family == "ml"`.
- [x] El router despacha a ML sin importar LightGBM ni Optuna (AST).
- [x] Payload con hiperparametros y metrica; router anota `policy_version`.
- [x] Errores de datos -> `SelectionContractError` con causa encadenada.
- [~] Es la primera `SelectionStrategy` concreta del repo (la clasica no tiene
  wrapper de router); se probo con `SelectionRouter` y estrategias falsas para
  las demas familias.

**Archivos:** `optimizadores/modelos_machine_learning/estrategia.py`,
`tests/optimizacion/modelos_machine_learning/test_estrategia.py`.

---

## TASK-MLS-6.0-C1 — Pruebas de integracion y regresion

**Estado: implementada.**

- [x] 3 clases que la politica 2.2 asigna a ML; `lumpy` no se enruta a ML.
- [x] Determinismo por seed, reanudacion, warm start, poda con `motivo`.
- [x] AST: sin bucle ask/tell propio ni `import optuna` en la familia ML.
- [x] Pruebas lentas con `@pytest.mark.slow`; `ruff` y `pyright` limpios en rutas nuevas.
- [x] Cobertura 92-100 % de los modulos nuevos (rcfile temporal).

**Archivos:** `tests/optimizacion/modelos_machine_learning/`.

---

## TASK-MLS-6.0-C2 — Benchmark vs baseline y ahorro de ASHA

**Estado: implementada, con una salvedad.**

- [x] En MAE medio por clase la seleccion supera al default y al ingenuo
  estacional en las 3 clases; gana en 8/12 series a cada uno.
- [x] ASHA reduce el costo 48.6 % en promedio (minimo 40.0 %) >= 30 %.
- [x] Metodo y resultados reproducibles con semilla fija (README).
- [~] **Salvedad:** en `intermittent` la ganancia sobre el default es ~2 % y
  solo gana 1 de 4 series por separado. Series sinteticas; falta validar con
  datos reales.

**Archivos:** `optimizadores/modelos_machine_learning/benchmarks/`,
`tests/optimizacion/modelos_machine_learning/test_benchmark_ml.py`.

---

## TASK-MLS-6.0-D1 — Documentacion

**Estado: implementada localmente; quedan pasos manuales en Notion.**

- [x] README, API_SPECIFICATION y TASKS_2.5 en
  `docs/features/2.Seleccion-config-modelos/2.5-MLSelectionStrategy/`.
- [x] ADR-014 y `GUIA_CICLO_VIDA_CORRIDA.md` actualizados.
- [ ] **Manual en Notion:** subsecciones y ADR de la pagina "6. MLSelectionStrategy"
  (hoy vacias); `Implementation Status` a `Completado` al cierre; ADR de
  `Propuesto` a `Aceptada` al merge.
