# ADR-017: Ejecución del Módulo 2 con reserva del 20 %, unidades paralelas aisladas y corrida persistida

**Date:** 2026-10-02
**Status:** Accepted
**Notion ADR:** aplica ADR-03-003 (reserva cronológica del 20 %) y ADR-03-004
(manifiesto tipado de candidatos); implementa 3.0 (recuperación por unidad) y
2.9 (control de reanudación) en la ruta del pipeline.
**Notion Task:** 3.1 (reserva temporal y causalidad), 3.2 (adaptador de candidatos)

## Context

El pipeline coordinado (L1-L4) corría L2 y L3 de forma secuencial, SKU por SKU,
y la primera falla de cualquier SKU detenía toda la etapa. L2 usaba la serie
completa, sin la reserva cronológica de ADR-03-003. Los modelos ajustados solo
existían en memoria y el resultado de una corrida no quedaba persistido. El
paralelismo por procesos existía en `seleccion_hpo.seleccionar_panel`, pero
el pipeline no lo usaba, y la reanudación de 2.9 solo se activaba si el llamador
configuraba cada estrategia a mano.

Una corrida de tesis sobre la semilla de la Fase 0 tiene 110 SKU y entre 1 y 4
familias por SKU. Necesita usar todos los núcleos de la máquina, terminar
aunque un SKU falle y poder retomarse sin repetir el HPO ya terminado.

## Decision

1. **Reserva (ADR-03-003, 3.1).** `ReserveCut` calcula un corte unificado t*:
   los últimos `ceil(0.2 × días del calendario del panel)` días quedan
   reservados para M3. L2 y L3 solo leen observaciones con fecha ≤ t*. Un SKU
   sin historia admisible suficiente, con huecos de calendario, con demanda no
   finita o cuya serie termina antes de t* se excluye con su causa registrada.
2. **Unidades aisladas (3.0).** Cada par SKU × familia es una unidad de L2
   (selección con HPO, ajuste final y pronóstico) y cada candidato es una
   unidad de L3 (walk-forward). `comun.ejecucion_paralela` corre las unidades
   en un pool `spawn` con BLAS de una hebra. Una unidad que falla queda
   registrada como `fallida` con su error; la etapa solo falla si ninguna
   unidad produce resultado. Cada unidad guarda pid, inicio y fin.
3. **Router en dos pasos.** `SelectionRouter.plan` devuelve las decisiones de
   la política y `execute` corre una sola; `route` conserva su comportamiento.
4. **Pronóstico en lugar del modelo ajustado.** Cada candidato entrega el
   pronóstico de los días reservados desde t*. El modelo ajustado no viaja
   entre procesos (un SARIMA serializado ocupa unos 40 MB) y M3 lo reconstruye
   desde la configuración, como pide ADR-03-004.
5. **Manifiesto tipado (ADR-03-004).**
   `optimizacion.manifiesto_candidatos` define la unión discriminada por
   familia, con `extra="forbid"` y todos los hiperparámetros obligatorios.
6. **Corrida persistida.** `pred-engine run` escribe
   `{data_root}/runs/{run_id}/` con `corrida.json`, `candidatos.json`,
   `pronosticos.parquet`, `evaluacion.parquet`, `walk_forward.parquet`,
   `unidades.jsonl` y `hpo/`. El `run_id` se deriva de la huella de la entrada,
   de las opciones y del código fuente. Con el mismo comando, los estudios HPO
   terminados se reconstruyen desde sus manifiestos (2.9) y no se reentrenan.
7. **Entrada desde la semilla.** `pred-engine run --seed-csv` ejecuta la Fase 0
   (opciones `--m0-*`) y entrega su panel a L1: M0 → M1 → M2 en un comando.
8. **Códigos de salida.** 7 = L4 ausente y sin fallos; 8 = L4 ausente con
   unidades fallidas aisladas; 1 = una etapa sin resultados.

## Rationale

La unidad SKU × familia no comparte estado con otras unidades: el resultado en
paralelo es idéntico al secuencial, como prueba la suite. Aislar fallas por
unidad es lo que exige 3.0 y evita que un SKU degenerado anule una corrida de
horas. Devolver el pronóstico y no el modelo mantiene pequeño el tráfico entre
procesos y respeta la decisión de ADR-03-004 de no serializar modelos.

## Consequences

- Evidencia real (semilla Kaggle, 110 SKU, 22 procesos): 212 unidades de L2 y
  hasta 144 de L3 corren con concurrencia máxima 22 y una aceleración medida de
  16,6 a 18,5 veces respecto del tiempo de cómputo sumado.
- `sku_class` llega de M1, que clasifica el panel completo, incluida la
  reserva. La ruta de familias de un SKU puede depender de datos posteriores a
  t*. Reclasificar sobre la historia admisible es una decisión pendiente.
- L3 sigue midiendo sobre la misma historia que usó el HPO
  (`selection_scope="same_history"`); la evaluación sobre la reserva es de M3.
- Si un proceso muere (por ejemplo, por memoria), las unidades sin terminar se
  registran como fallidas y una nueva ejecución del mismo comando las retoma.
- Cambiar el código cambia el `run_id`; para retomar una corrida interrumpida
  hay que usar el mismo código y el mismo comando, o `--run-id`.

## Related

- ADR-010 y ADR-011 (control de corrida y checkpoints del HPO).
- ADR-013 (ASHA secuencial dentro de cada estudio; el paralelismo es entre
  estudios).
- `src/pred_engine/pipeline.py`, `pipeline_setup.py`, `run_artifacts.py`,
  `cli.py`, `comun/ejecucion_paralela.py`,
  `optimizacion/manifiesto_candidatos.py`.
