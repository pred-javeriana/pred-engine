# Ejecución M0 → M1 → M2

Guía de operación de la ruta completa: la Fase 0 genera el panel sintético
(M0), L1 lo ingiere y clasifica (M1) y L2-L3 seleccionan, ajustan y pronostican
cada SKU (M2). Las decisiones de diseño están en ADR-016, ADR-017, ADR-018,
ADR-019 y ADR-020.

## Comando

```bash
uv sync --extra dev
uv run pred-engine run --seed-csv inventory_data.csv --data-root data \
  --m0-metodo mbb-directo \
  --m0-columna sku_id=Item_ID --m0-columna timestamp=Date \
  --m0-columna demand_qty=Avg_Usage_Per_Day \
  --m0-columna lead_time_days=Restock_Lead_Time
```

`inventory_data.csv` es el inventario del conjunto Kaggle "Hospital Supply
Chain" (semilla 0.1 del hub). Otras entradas posibles:

- `--csv panel.csv`: un panel canónico ya generado (por ejemplo, la salida de
  `pred-engine-fase0`).
- `--parquet data/processed/panel.parquet`: un artefacto 1.4 ya clasificado.

## Qué hace cada módulo

1. **M0 (Fase 0).** Lleva la semilla a calendario diario, genera
   `--m0-n-series` réplicas por SKU con el método elegido, aplica las leyes
   físicas, exige paridad del 5 % en media y varianza y deposita
   `data/raw/panel_sintetico_fase0.csv` (WORM, 0444) con su bitácora en
   `data/logs/`. Repetir el comando reutiliza el artefacto si la configuración
   no cambió; con otra configuración falla por WORM y hay que usar otro
   `--m0-nombre` u otro `--data-root`.
2. **M1 (L1).** Valida el contrato, clasifica cada SKU (Syntetos-Boylan) con
   su historia hasta t*, sin los días reservados (ADR-019), y publica
   `data/processed/panel_sintetico_fase0.parquet` (contrato 1.4).
3. **M2 (L2-L3).** Calcula el corte t* (los últimos `ceil(0.2 × días)` días
   quedan reservados para M3), excluye con causa los SKU no elegibles y corre
   cada SKU × familia como unidad: HPO (TPE + ASHA sobre walk-forward), ajuste
   final en la historia admisible y pronóstico de los días reservados. El HPO
   solo elige configuraciones que ajustan en todas las ventanas, y un
   pronóstico final degenerado (regla #3) hace fallar su unidad (ADR-020). L3
   vuelve a evaluar cada candidato con walk-forward causal.

## Opciones de M2

| Opción | Por defecto | Efecto |
| --- | --- | --- |
| `--families` | `classical ml dl` | Subconjunto de la matriz de enrutamiento; `foundation` requiere el extra `foundation` y los pesos de Chronos-2. |
| `--trials` | presupuesto por perfil | Trials de HPO por estudio; con presupuestos chicos algunos SKU pueden quedar sin trial completo. |
| `--workers` | núcleos - 1 | Procesos para las unidades de L2 y L3. |
| `--min-train`, `--horizon`, `--step`, `--seasonality`, `--metric` | 40, 7, 7, 7, mase | Ventanas walk-forward del HPO y de L3. |
| `--seed` | 0 | Semilla del HPO y de los modelos. |
| `--runs-dir`, `--run-id` | `data/runs`, derivado | Ubicación e identidad de la corrida. |

Las opciones de la Fase 0 son las de `pred-engine-fase0` con el prefijo
`--m0-` (por ejemplo, `--m0-metodo`, `--m0-n-series`, `--m0-seed`).

## Artefactos de la corrida

Todo queda en `data/runs/{run_id}/`:

| Archivo | Contenido |
| --- | --- |
| `corrida.json` | Entradas (semilla, artefacto M0, Parquet M1, hashes y filas), opciones, resumen de etapas, corte t*, exclusiones, fallas, evidencia de paralelismo, tiempo de pared, versiones de librerías y la procedencia de cada candidato (perfil, política, referencia al estudio HPO y su evidencia). |
| `candidatos.json` | Manifiesto de candidatos para M3 (ADR-03-004), el que valida `forecasting.adaptador_candidatos`: contexto de partición (huella del Parquet de M1, t* y fracción reservada), y configuración completa, semilla e identidad de cada candidato. |
| `pronosticos.parquet` | Pronóstico diario de cada candidato para los días reservados (`sku_id`, `sku_class`, `familia`, `modelo`, `timestamp`, `pronostico`). |
| `evaluacion.parquet` | Métrica agregada walk-forward y medias de MAE, RMSE, sMAPE y MASE por candidato. |
| `walk_forward.parquet` | Valor real y pronóstico de cada día de cada ventana walk-forward. |
| `unidades.jsonl` | Una línea por unidad de L2 o L3: estado, pid, inicio, fin, segundos y error. |
| `hpo/` | Manifiesto y trials de cada estudio HPO (control de reanudación 2.9). |

El `run_id` se deriva del hash de la entrada, de las opciones y de la huella
del código fuente. El mismo comando con el mismo código siempre apunta a la
misma corrida.

## Paralelismo

Cada unidad registra su pid y su intervalo. `corrida.json` resume, por etapa,
los procesos distintos, la concurrencia máxima observada y la aceleración
(tiempo de cómputo sumado dividido por el tiempo de pared del tramo). Los
procesos se crean con `spawn` y BLAS de una hebra, así que cada unidad usa un
núcleo.

## Fallas y reanudación

- Una unidad que falla (por ejemplo, un estudio sin trials completos) queda en
  `unidades.jsonl` y en `failures`; el resto de la etapa sigue.
- Repetir el mismo comando retoma la corrida: los estudios HPO terminados se
  reconstruyen desde su manifiesto sin reentrenar, los interrumpidos continúan
  desde su último checkpoint y solo los que faltan arrancan de cero. L3 y los
  pronósticos se recalculan, porque son deterministas.
- Un estudio en estado `fallida` no se reabre (2.9); su unidad vuelve a fallar
  con ese motivo.
- Si cambia el código, cambia el `run_id`. Para continuar una corrida anterior
  hay que usar el mismo código o pasar `--run-id` de forma explícita.

## Códigos de salida

| Código | Significado |
| --- | --- |
| 7 | L1-L3 completas sin fallas; L4 no existe en el repositorio. |
| 8 | L1-L3 completas con unidades fallidas aisladas; L4 no existe. |
| 1 | Error de entrada o de la Fase 0, o una etapa sin ningún resultado. |
| 0 | Solo con un validador L4 inyectado por el llamador. |

## Referencia en la máquina de la tesis

24 núcleos, 30 GB de RAM y `--workers 22`. Semilla Kaggle con `mbb-directo` y
presupuesto por defecto:

- M0: 53 482 filas, 110 SKU (10 de la semilla y 100 sintéticos), ~90 % de días
  sin demanda, igual que la semilla.
- M1: con la historia hasta t*, 98 SKU `intermittent` y 12 `lumpy`;
  t* = 2025-11-04 y 100 días reservados.
- M2: 208 unidades de L2 (110 clásicas y 98 ML), 203 completas y 5 clásicas
  fallidas de forma aislada: 3 sin ningún trial que ajuste en todas las
  ventanas y 2 con pronóstico final nulo (regla #3). Las 203 unidades de L3
  completas. Tiempo de pared de 2 min 23 s, concurrencia máxima 22 y
  aceleración de 20,1 (L2) y 18,2 (L3). Código de salida 8.
- Ninguna ventana walk-forward supera 1,12 veces el máximo histórico de su
  SKU hasta t*, ningún pronóstico final lo supera (máximo 0,56 veces) y
  ninguno es nulo.
- El adaptador de M3 acepta los 203 candidatos de `candidatos.json` (105
  SARIMA, 80 de ellos con constante, y 98 LightGBM). Cada candidato
  reconstruido desde el manifiesto y ajustado con la historia admisible
  reproduce exactamente el pronóstico de M2.
- Repetir el comando reconstruye los estudios terminados en menos de 2 s de
  L2, no reabre los 3 estudios fallidos, no modifica ningún archivo de `hpo/`
  y escribe los mismos candidatos y el mismo contexto en `candidatos.json`;
  solo cambia `emitido_en`.

## Límites conocidos

- Con la semilla Kaggle la política no enruta DL: DL solo aplica a SKU
  `smooth` y `erratic`.
- La política 2.2 enruta los SKU `lumpy` solo a la familia clásica. Si esa
  unidad falla, el SKU queda sin candidato: 2 de 110 con la semilla Kaggle
  (`100::syn009` y `105::syn005`). Elegir el siguiente trial elegible cuando
  el ganador falla en L2 es una decisión pendiente (ADR-020).
- La familia `foundation` queda fuera de las corridas de referencia: necesita
  el extra `foundation` y los pesos de Chronos-2, que no están instalados en
  la máquina de la tesis.
- L4 (validación retrospectiva) y la evaluación de M3 sobre la reserva no
  existen todavía; la corrida entrega lo que M3 necesita para hacerla.
