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
5. **Manifiesto de candidatos (ADR-03-004).** `candidatos.json` usa la
   definición única del contrato M2 → M3
   (`comun.modelos.manifiesto_candidatos`) y se construye con
   `optimizacion.router.construir_manifiesto`, el mismo emisor que valida el
   adaptador de M3 (`forecasting.adaptador_candidatos`, 3.2). El contexto de
   partición es el real de la corrida: huella SHA-256 del Parquet de M1, t* y
   la fracción reservada. El manifiesto lleva solo lo que M3 necesita para
   reconstruir cada candidato (configuración completa, semilla e identidad);
   la procedencia (perfil, política, referencia al estudio HPO y su
   evidencia) y las versiones de librerías quedan en `corrida.json`.
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
- `sku_class` llegaba de M1, que clasificaba el panel completo, incluida la
  reserva. ADR-019 resuelve la fuga: M1 clasifica con la historia hasta t*.
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
  `cli.py`, `comun/ejecucion_paralela.py`.
- `src/pred_engine/comun/modelos/manifiesto_candidatos.py`,
  `optimizacion/router/manifiesto.py` y `forecasting/adaptador_candidatos/`
  (contrato, emisor y validación del handoff M2 → M3).
