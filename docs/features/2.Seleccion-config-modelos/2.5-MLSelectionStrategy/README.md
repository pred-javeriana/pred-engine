# Seleccion 2.5 — MLSelectionStrategy (HPO sobre LightGBM)

## Que se hizo

Implementacion de las tareas `TASK-MLS-6.0-*` de "6. MLSelectionStrategy"
(spec 2.5 Seleccion de Machine Learning). Es un **wrapper delgado**: define el
espacio de hiperparametros de ML y delega toda la optimizacion en el motor de
HPO de 2.4 (TPE + ASHA + Walk-Forward). Decisiones en ADR-014.

1. Modelo `LightGBMForecaster` (regresion autoregresiva, pronostico
   recursivo) y `fabrica_ml`.
2. Espacio de busqueda `EspacioML` / `construir_espacio_ml`.
3. Nucleo compartido `seleccion_hpo.py` (validacion, estudio por SKU, panel
   paralelo), extraido de `classical_selection.py`.
4. `seleccionar_configuracion_ml` / `seleccionar_por_panel_ml`.
5. `MLSelectionStrategy` (`family="ml"`) registrable en el router 2.1/2.2.
6. Benchmark contra baselines y ahorro de ASHA.

**Alcance por clase de SKU:** la politica 2.2 enruta `ml` solo a `smooth`,
`erratic` e `intermittent`. `lumpy` no llega a esta familia.

## Como

```
SelectionRouter ──► MLSelectionStrategy.select(request, profile)
                          │  serie ordenada por timestamp; presupuesto por perfil
                          ▼
              seleccionar_configuracion_ml(...)        (ml_selection.py)
                          │  EspacioML + fabrica_ml
                          ▼
              seleccion_hpo.seleccionar_con_hpo(...)   (nucleo compartido)
                          ▼
              HPO.estudio.ejecutar_estudio(...)        (TPE + ASHA + Walk-Forward, 2.4)
                          ▼
              fabrica_ml ──► LightGBMForecaster.fit / predict (por ventana)
```

| Pieza | Responsabilidad | Archivo |
| --- | --- | --- |
| `LightGBMForecaster`, `fabrica_ml`, `construir_muestras` | Modelo autoregresivo determinista | `comun/modelos/modelos_machine_learning/lgbm.py` |
| `SerieCortaError`, `AjusteModeloError` | Errores de ajuste | `comun/modelos/modelos_machine_learning/errores.py` |
| `EspacioML`, `construir_espacio_ml`, `min_train_recomendado_ml` | Espacio de hiperparametros | `optimizadores/modelos_machine_learning/espacio_ml.py` |
| `seleccionar_configuracion_ml`, `seleccionar_por_panel_ml` | Wrapper hacia HPO | `optimizadores/modelos_machine_learning/ml_selection.py` |
| `MLSelectionStrategy`, `PresupuestoHPO`, `PRESUPUESTO_POR_PERFIL` | Estrategia del router | `optimizadores/modelos_machine_learning/estrategia.py` |
| `ConfiguracionSeleccionadaML`, `ResultadoSeleccionML` | Salida de la seleccion | `comun/dataclasses/modelos_machine_learning.py` |
| `exigir_serie`, `seleccionar_con_hpo`, `seleccionar_panel`, `series_por_sku` | Nucleo comun a ML y DL | `optimizadores/seleccion_hpo.py` |
| `serie_sintetica`, `comparar_con_baseline`, `ahorro_asha` | Benchmark | `optimizadores/modelos_machine_learning/benchmarks/` |

## Resultados de benchmark (TASK-MLS-6.0-C2)

Series sinteticas por clase (n=200, m=7, semillas 0-3), 25 trials,
`ReglasPoda(min_ventanas=4, factor_reduccion=2)`. Se aparta el ultimo tramo de
28 dias, se selecciona sobre lo anterior y se mide MAE del holdout.
Reproducible con:

```bash
uv run pytest tests/optimizacion/modelos_machine_learning/test_benchmark_ml.py -m slow -q
```

| Clase | MAE seleccionada | MAE default | MAE ingenuo estacional | Gana al default / al ingenuo | Ahorro ASHA |
| --- | --- | --- | --- | --- | --- |
| smooth | 1.52 | 1.88 | 1.75 | 4/4 · 3/4 | 48.9 % |
| erratic | 1.10 | 1.26 | 1.31 | 3/4 · 2/4 | 46.6 % |
| intermittent | 5.24 | 5.35 | 5.54 | 1/4 · 3/4 | 50.4 % |
| **Total** | | | | **8/12 · 8/12** | **48.6 % (min 40.0 %)** |

Lectura honesta: en MAE medio por clase la seleccion supera al default y al
ingenuo en las tres clases, y ASHA supera con holgura el 30 % de ahorro
exigido (proxy: ventanas Walk-Forward evaluadas, igual que en 2.4). En
`intermittent` la ganancia sobre el default es pequena (~2 %) y solo gana 1
de 4 series por separado. Son series sinteticas; falta validar con datos
reales.

## Limitaciones conocidas

- **Sin calendario real:** el Walk-Forward entrega `y` 1D sin timestamps; la
  estacionalidad entra como fase `posicion mod m` (`m=7` por defecto).
- **Panel paralelo (`n_procesos > 1`) usa `spawn`:** el script que lo llame
  debe llevar `if __name__ == "__main__":` en todos los sistemas operativos
  (con `fork` LightGBM se cuelga; ver ADR-014).
- **La restriccion de costo consume presupuesto:** una configuracion que la
  viola se marca `fallido` (`configuracion_invalida`).
- **`seleccionar_por_panel_ml` rechaza `run_id`** (colisionaria entre SKUs);
  para persistencia por SKU usar `MLSelectionStrategy(raiz_corrida=...,
  sesion=...)`, que deriva `ml-{sku_id}-{sesion}`.
- **Reanudacion con `TPESampler`:** hereda la limitacion documentada en
  ADR-012.

## Que reutilizara 2.6 (Deep Learning)

`seleccion_hpo.py` (`exigir_serie`, `seleccionar_con_hpo`,
`seleccionar_panel`), el patron de `EspacioX` + `construir_espacio_x` +
`fabrica_x` y el de estrategia con `PresupuestoHPO`. 2.6 solo aporta su
espacio (arquitectura + entrenamiento), su fabrica y su estrategia.

## Verificacion

```bash
uv run pytest tests/optimizacion/modelos_machine_learning tests/comun/modelos/test_lgbm_forecaster.py -q
uv run pytest tests/optimizacion/modelos_machine_learning -m slow -q
uv run pytest tests/optimizacion tests/comun -q          # sin regresiones (HPO, router, clasicos)
uv run ruff check src tests
uv run pyright
```

Cobertura de los modulos nuevos (rcfile temporal, porque `pyproject.toml`
omite `optimizadores/*`): `lgbm.py` 97 %, `espacio_ml.py`, `ml_selection.py`,
`estrategia.py` 100 %, `seleccion_hpo.py` 92 %.
