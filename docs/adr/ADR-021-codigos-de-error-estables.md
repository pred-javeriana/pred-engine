# ADR-021: Códigos de error estables por tipo de excepción y por causa de exclusión

**Date:** 2026-10-19
**Status:** Proposed
**Notion ADR:** pendiente de registrar.
**Notion Task:** TASK-UI-1.0-A3-G6 (catálogo y mapeo de excepciones), TASK-UI-1.0-A3-G7
(código por exclusión y falla de unidad; RF-MOD-13)

## Context

La interfaz necesita saber por qué falló o se omitió una unidad para mostrarlo en español con la
acción correctiva (RNF-USA-02). El motor ya aislaba los fallos y excluía SKU con causa, pero la
causa era texto libre (`NombreDeClase: mensaje`), y dos casos no tenían tipo propio (el
`ValueError` de `extract_csv` y «ningun trial de HPO completo»). La plataforma tenía que
interpretar texto. Solo la ingesta tenía una tabla de códigos (`ingesta/informe.py`).

## Decision

1. Un registro único, `pred_engine.comun.errores_codigo`, deduce el código del **tipo** de la
   excepción (`codigo_de`), nunca del mensaje. Las clases se resuelven de forma perezosa por
   (módulo, nombre) para que `comun` no importe capas superiores ni cree ciclos y para que sirva
   en los subprocesos `spawn`. Lo que no tiene tipo propio cae en `pipeline.stage_failed`, o en
   `evaluation.failed` si la unidad era de L3.
2. Los dos casos genéricos reciben tipo propio como subclases, para no romper a quien ya los
   captura: `ArchivoNoCsvError` y `RutaFueraDeRawError` (`ValueError`) y
   `NingunTrialCompletoError` (`SelectionContractError`).
3. Cada exclusión de SKU antes de L2 lleva un código del grupo `eligibility.*`
   (`SkuExclusion.code`); los números de la causa siguen en el texto.
4. Los códigos se escriben como campo **opcional** junto al texto (`code`) en `corrida.json`
   (`exclusions`, `failures`), `unidades.jsonl`, `progreso.jsonl` (`excluded`, `unit_finished`,
   `stage_finished` fallida) y `seleccion.jsonl` (`no_ejecutable`, `error`). El formato
   `NombreDeClase: mensaje` no cambia, así que los esquemas `pred-engine.*/1` siguen vigentes y
   los lectores anteriores ignoran el campo.

## Consequences

- La plataforma guarda el código del motor tal cual; para corridas anteriores deduce el código de
  la clase o del texto.
- Una clase nueva de error debe añadirse a la tabla: `tests/comun/test_errores_codigo.py` falla si
  una excepción tipada de un módulo `errores` queda sin código.
- Si la etapa entera falla, `corrida.json` no lleva `failures`; el código va en el evento
  `stage_finished`.

## Alternatives Considered

- **Atributo `codigo` en cada excepción.** Obliga a tocar todas las clases y no cubre las de la
  biblioteca estándar; la tabla centraliza y se prueba.
- **Deducir el código en la plataforma.** Es lo que se hacía y exige interpretar texto.

## Related

ADR-020 (elegibilidad), sección 11 del contrato frontend-motor.
