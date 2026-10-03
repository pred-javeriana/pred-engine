# ADR-016: Método de aumento explícito en la Fase 0 (enmienda a ADR-01-007)

**Date:** 2026-10-02
**Status:** Accepted
**Notion ADR:** enmienda a ADR-01-007 (STL + MBB de residuales); la página de
Notion no se modificó y queda pendiente registrar la enmienda allí.
**Notion Task:** 0.2 (MBB), 0.3 (compuerta de restricciones), 0.4 (orquestación)

## Context

La semilla designada para la Fase 0 es el inventario del conjunto Kaggle
"Hospital Supply Chain" (`inventory_data.csv`): 10 ítems con unos 50 registros
de consumo cada uno, repartidos en unos 490 días de calendario. Con los
nombres de columna del archivo (`Item_ID`, `Date`, `Avg_Usage_Per_Day`,
`Restock_Lead_Time`), la semilla es transaccional e intermitente: cerca del
90 % de los días no tiene demanda.

Sobre esa semilla, `pred-engine-fase0` terminaba siempre con
`DivergenceRejectionExhausted` y una traza de Python. Se verificaron dos
causas:

1. **Calendario comprimido.** Cada fila se trataba como un día consecutivo,
   así que 490 días intermitentes se convertían en 50 días densos.
2. **Método.** ADR-01-007 recombina tendencia y estacionalidad STL con un MBB
   de los residuales. En la semilla llevada a calendario diario, esa
   recombinación llena los días sin demanda con valores positivos pequeños: el
   ADI baja de ~10 a ~1.7 (la serie deja de ser intermitente) y, después de las
   leyes físicas de 0.3, ninguna de 300 candidatas por SKU pasa la compuerta de
   paridad del 5 %.

La especificación 0.2 del hub describe en cambio un MBB directo de la serie:
bloques contiguos que conservan las rachas de cero, ruido solo sobre la
demanda positiva y recorte a cero.

## Decision

1. El método de aumento es una elección explícita por corrida
   (`--metodo`), registrada en la bitácora de la Fase 0:
   - `stl-mbb` (valor por defecto): ADR-01-007 sin cambios de método.
   - `mbb-directo`: MBB directo de 0.2 con bloques de 30 días y ruido
     gaussiano del 3 % de la media positiva solo en días con demanda.
2. Las corridas de tesis con la semilla Kaggle usan `mbb-directo`.
3. Cambios comunes a ambos métodos:
   - La semilla se lleva a un calendario diario por SKU: las transacciones del
     mismo día se suman, los días sin registro tienen demanda 0 y el lead time
     vigente se propaga.
   - Las leyes físicas (demanda entera no negativa) se aplican a cada
     candidata **antes** de la compuerta de paridad, porque esa es la serie
     que llega al artefacto.
   - El presupuesto de reintentos sube de 20 a 200 y cada SKU usa una semilla
     aleatoria propia derivada de la semilla de la corrida.
   - `--columna CANONICA=ORIGEN` mapea columnas de la semilla de forma
     explícita.
   - El artefacto queda en modo 0444, la bitácora guarda método, mapeo,
     configuración y huella, y `--reutilizar` reconoce una corrida idéntica ya
     depositada en lugar de fallar por WORM.
   - Los errores esperables se informan en una línea, sin traza.

## Rationale

Mantener `stl-mbb` como valor por defecto respeta ADR-01-007 para semillas
densas con tendencia y estacionalidad semanal. El MBB directo es el único de
los dos métodos que conserva la intermitencia de la semilla designada, que es
la propiedad que el Módulo 1 mide para enrutar cada SKU. Hacer explícito el
método evita que una corrida de tesis dependa de un valor por defecto
implícito.

## Consequences

- Corrida real sobre la semilla Kaggle (`--metodo mbb-directo`, 10 series por
  ítem): 53 482 filas, 110 SKU (10 de la semilla y 100 sintéticos), tasa de
  rechazo del 86 % y 0,75 s. La fracción de días sin demanda es 89,7 % en
  promedio (la semilla tiene ~90 %), el ADI medio es 10,0 y ningún SKU queda
  sin demanda. El Módulo 1 acepta el CSV sin cambios y clasifica 102 SKU como
  `intermittent` y 8 como `lumpy`.
- Con esta semilla la política del Módulo 2 nunca enruta la familia DL
  (DL solo se habilita para `smooth` y `erratic`).
- ADR-01-007 sigue vigente para `stl-mbb`; la enmienda que limita su uso a
  semillas no intermitentes debe registrarse en Notion.

## Related

- ADR-01-007 (Notion): STL + MBB de residuales.
- `src/pred_engine/aumentacion/fase0.py`, `rechazo.py`, `bitacora.py`, `worm.py`.
- `docs/features/0.3-0.4-simulacion-fase0/`.
