# ADR-014: LightGBM como modelo base de ML, con features autoregresivos y seleccion por HPO

**Date:** 2026-09-21
**Status:** Proposed
**Notion ADR:** por crear en "ADRs Pendientes" de 6. MLSelectionStrategy (numero ADR-02-0XX segun el registro)
**Notion Task:** TASK-MLS-6.0-A1 / B1 / B2 / B3 / B4

## Context

La especificacion 2.5 (Seleccion de Machine Learning, tarea "6. MLSelectionStrategy")
establece que los modelos de ML no tienen una parametrizacion equivalente a
`p,d,q`, asi que la complejidad se selecciona por Hyperparameter
Optimization sobre validacion temporal (Walk-Forward), no por AICc. Los
hiperparametros que cita: `n_estimators`, `max_depth`, `learning_rate`,
`min_child_weight`, `subsample`, `colsample` y regularizacion.

El motor de HPO (TPE + ASHA + Walk-Forward, ADR-012/013) ya existe y es
agnostico de familia. Faltaba: (1) el modelo concreto, (2) su espacio de
busqueda, (3) el envoltorio que lo conecta al router. Ademas la politica 2.2
(`router/politica.py::FAMILIES_BY_SKU_CLASS`) enruta la familia `ml` solo a
`smooth`, `erratic` e `intermittent`; **`lumpy` queda fuera**.

## Decision

1. **Modelo: LightGBM** (`lightgbm>=4.0`), usado por su API nativa
   (`lightgbm.train`), sin scikit-learn. Cubre exactamente los hiperparametros
   de 2.5 (`min_child_weight` -> `min_sum_hessian_in_leaf`, `colsample_bytree`
   -> `feature_fraction`, `subsample` -> `bagging_fraction`,
   `reg_alpha/lambda` -> `lambda_l1/l2`). Se descartaron `HistGradientBoosting`
   de scikit-learn (no tiene `subsample` ni `min_child_weight`) y XGBoost
   (mas pesado y mas lento para muchos ajustes pequenos de Walk-Forward).

2. **Determinismo:** `seed` fijo, `num_threads=1`, `deterministic=True`,
   `force_row_wise=True`. Mismo `seed` + misma configuracion + misma serie =
   mismo pronostico (RNF-REP-01/02); probado en `test_lgbm_forecaster.py`.

3. **Features (regresion autoregresiva):** por muestra, las ultimas `lags`
   observaciones, tres agregados de esa misma ventana (media, desviacion,
   fraccion de ceros) y, si `m > 1`, la fase estacional `posicion mod m`.
   **No hay calendario real**: el Walk-Forward entrega `y` como arreglo 1D
   sin timestamps (contrato `Pronosticador`). La fase estacional es
   consistente entre entrenamiento y prediccion porque las ventanas de
   Walk-Forward son expansivas desde el indice 0.

4. **Pronostico multi-paso recursivo:** cada prediccion se reinyecta como
   observacion. Es el unico esquema que cabe en `predict(horizon)` sin un
   modelo por horizonte; el "directo" multiplicaria el costo de cada trial
   por `horizonte`. La demanda pronosticada se recorta a >= 0 dentro del lazo.

5. **Espacio de busqueda** (`espacio_ml.py`, `EspacioML`): `lags` (3-14),
   `n_estimators` (20-300), `max_depth` (2-8), `learning_rate` (0.01-0.3, log),
   `min_child_weight` (1-20, log), `subsample` (0.5-1), `colsample_bytree`
   (0.5-1), `reg_alpha` y `reg_lambda` (1e-8-10, log). `lags` cumple el papel
   de `p,d,q`: es el parametro de capacidad de la entrada. Restriccion de
   costo `n_estimators * max_depth <= 1500`.

6. **Presupuesto por perfil topologico** (`estrategia.py::PRESUPUESTO_POR_PERFIL`):
   `dense_stable` 30 trials, `dense_variable` 40, `sparse_stable` 25,
   `sparse_variable` 25; `ReglasPoda` por defecto. Inyectable.

7. **Wrapper delgado:** `seleccionar_configuracion_ml` solo define espacio y
   fabrica y delega en `seleccion_hpo.seleccionar_con_hpo` ->
   `ejecutar_estudio`. **No** hay bucle ask/tell ni `import optuna` en la
   familia ML (verificado por AST en `test_delegacion.py`).

8. **Nucleo compartido para ML y DL:** `optimizadores/seleccion_hpo.py`
   concentra lo que las familias hacen igual (`exigir_serie`,
   `seleccionar_con_hpo`, `series_por_sku`, `seleccionar_panel`). Clasicos
   se migro a el sin cambiar su comportamiento; 2.6 (Deep Learning) lo
   reutilizara.

9. **Panel paralelo con `spawn`, no `fork`.** El pool de procesos se crea con
   el contexto `spawn`. Con `fork`, si el proceso padre ya entreno un modelo
   LightGBM, el hijo se cuelga (el runtime de OpenMP no sobrevive a `fork`);
   se reprodujo fuera de pytest. Costo: los hijos reimportan el paquete al
   arrancar, y todo script con `n_procesos > 1` debe llevar
   `if __name__ == "__main__":` en cualquier sistema operativo.

## Rationale

- Reutiliza el motor de HPO existente: 90 % de reutilizacion entre verticales.
- LightGBM cubre la spec sin recortar el espacio de busqueda.
- La estrategia solo depende de `router.contratos`; el router no importa la
  familia ML ni LightGBM (verificado por AST).

## Consequences

- Nueva dependencia runtime: `lightgbm>=4.0` (`pyproject.toml`, `uv.lock`).
- `lumpy` no se cubre por la familia ML: es decision de politica 2.2, no de
  esta estrategia. El warm start (`historico_previo`) queda expuesto pero
  opcional.
- La restriccion de costo rechaza configuraciones via trial `fallido`
  (`configuracion_invalida`), que consume presupuesto: en una corrida de
  prueba, 2 de 32 trials. Es el mismo mecanismo que usa clasicos.
- Sin calendario real, la estacionalidad depende de `m` (default 7). Series
  con otra periodicidad exigen otro `m` en `EspacioML`.
- La ganancia de la seleccion sobre el default es pequena en demanda
  intermitente (~2 % de MAE, ver README de 2.5): honesto y esperable, porque
  la metrica de seleccion (MASE en Walk-Forward) no garantiza el MAE de un
  holdout corto con muchos ceros.
- Riesgo abierto: solo se probo en series sinteticas; queda pendiente
  validar con datos reales del Modulo 1.

## Related

- ADR-012 (TPE sobre Optuna), ADR-013 (ASHA secuencial), ADR-008 (router).
- `docs/features/2.Seleccion-config-modelos/2.5-MLSelectionStrategy/`.
