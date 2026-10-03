# ADR-018: Familia clásica en demanda intermitente: constante sin diferenciación y ASHA sin competidores degenerados

**Date:** 2026-10-02
**Status:** Accepted
**Notion ADR:** complementa ADR-02-005 (ASHA) y la selección clásica de 2.4;
pendiente de registrar en Notion.
**Notion Task:** 2.4 (selección clásica con HPO)

## Context

Con el panel de la Fase 0 sobre la semilla Kaggle (102 SKU intermitentes y
8 lumpy, ~90 % de días sin demanda), la familia clásica no entregaba candidato
para la mayoría de los SKU: todos sus trials terminaban podados. Se verificaron
dos causas.

1. **SARIMA sin constante.** `SarimaForecaster` usaba `trend=None`. Un ARMA no
   diferenciado sin constante pronostica hacia cero; después de una racha de
   ceros el pronóstico recortado a cero es nulo y la poda semántica
   (`prediccion_nula`) descarta el trial. En el SKU `105`, un ARMA(1,1) tenía
   25 de 51 ventanas degeneradas sin constante y 0 de 51 con constante.
2. **ASHA comparaba contra trials degenerados.** Un trial que pronostica todo
   cero obtiene un MASE parcial muy bajo en demanda intermitente. Ese valor
   quedaba como competidor de su escalón aunque la regla #3 descartara el
   trial después, y ASHA podaba a las configuraciones válidas que pronostican
   el nivel medio.

## Decision

1. `fabrica_sarima` incluye una constante (`tendencia="c"`) cuando `d = D = 0`
   y ninguna (`"n"`) cuando hay diferenciación, como hace auto.arima
   (Hyndman y Khandakar, 2008) con la media de un modelo no diferenciado. Con
   diferenciación la constante sería una deriva, que no se incluye.
2. `ClassicalSelectionStrategy` publica `tendencia` en `forecast_config`, de
   modo que el manifiesto de candidatos (ADR-03-004) la declara de forma
   explícita y M3 no depende de una regla implícita. `ConfigSarima` la exige
   y rechaza la constante cuando `d` o `D` es mayor que cero.
3. `PodadorASHAOptuna` excluye de los competidores de un escalón a los trials
   cuyo motivo de cierre es poda semántica (`PREFIJO_PODA_SEMANTICA`).

## Rationale

Incluir la media en un modelo estacionario es la práctica estándar de
Box-Jenkins, no un cambio del espacio de búsqueda. La poda semántica existe
para que una configuración degenerada no gane por engañar a la métrica; dejar
que su valor parcial fije el umbral de ASHA contradecía esa misma regla.

## Consequences

- Corrida real con el presupuesto por defecto (semilla Kaggle, 110 SKU,
  22 procesos, antes de ADR-019 y ADR-020): 110 de 110 SKU obtienen candidato
  clásico y 102 de 102 obtienen candidato ML; en 80 de los 110 candidatos
  clásicos el HPO elige `d = D = 0` con constante. Ningún pronóstico desde t*
  es negativo ni no finito.
- Con presupuestos chicos (por ejemplo, `--trials 5`) todavía hay SKU sin
  trial completo: todos los SKU comparten la semilla del muestreador y los
  primeros trials pueden ser todos diferenciados. Esas unidades quedan como
  fallas aisladas (código de salida 8).
- Un candidato clásico (`104::syn000`, SARIMA(2,0,2)(2,0,0)7) tuvo ventanas
  walk-forward explosivas (~1e61) que la media recortada del 10 % ocultó en el
  valor agregado. ADR-020 identifica la causa, la estimación sin restricción,
  y la corrige.

## Related

- ADR-013 (ASHA como decisor puro).
- `src/pred_engine/optimizacion/optimizadores/modelos_clasicos/classical_selection.py`.
- `src/pred_engine/optimizacion/optimizadores/HPO/adaptador_optuna.py`, `poda.py`.
