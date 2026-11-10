# ADR-024: Hechos estructurados de los errores

**Date:** 2026-11-10
**Status:** Proposed
**Notion ADR:** pendiente de registrar.

## Context

Con el código estable (ADR-022) la interfaz sabe *qué* falló, pero el *porqué* seguía en el
texto. Una búsqueda sin ensayos completos (`selection.no_viable_trial`) explicaba sus causas en
`resumen_causas`: conteos y motivos como diccionarios de Python, con órdenes SARIMA, módulos de
raíces y el nombre calificado de una restricción. La plataforma solo podía mostrar ese texto tal
cual o interpretarlo.

## Decision

1. Una excepción que conoce los hechos de su fallo los lleva en `hechos`, con la forma de
   `ErrorFacts` del contrato 1.3.0 (sección 11.1 de `contrato-frontend-motor.md`): `search`
   (búsqueda sin ensayos completos), `fit` (ajuste que falló) e `history` (historia que faltó).
   `pred_engine.comun.hechos_error` es el único dueño de esa forma: `ErrorConHechos`, los
   constructores `ajuste` e `historia`, y `hechos_de`, que lee los hechos de una excepción o, si
   no los tiene, los de su causa (`raise ... from`).
2. Los hechos nacen donde nace el motivo: el modelo al rechazar un ajuste (`fit`), el ejecutor de
   la ventana (`fallo_hechos`), el estudio al cerrar un trial (atributo `hechos` del trial, que
   viaja con el estudio guardado y sobrevive a una reanudación) y ASHA al podar
   (`DecisionPoda.hechos`). `ResultadoEstudio.hechos_busqueda` los agrega: totales, cada causa con
   su conteo, de la más frecuente a la menos, y el primer ejemplo que registró hechos.
3. La restricción violada se nombra con `nombre_restriccion` (`orden_total_acotado`), sin módulo:
   la huella del espacio sigue usando la etiqueta calificada de siempre.
4. `RegistroUnidad.hechos` los lleva a `facts`, junto a `code`, en `unit_finished` y
   `stage_finished`, `unidades.jsonl`, `corrida.json` (`failures`) y `seleccion.jsonl` (`error`).
   Una unidad sin hechos conocidos no lleva el campo; el texto no cambia.

## Consequences

- La plataforma muestra causas, conteos y ejemplos sin leer el mensaje; el texto queda como
  detalle para soporte. Para corridas anteriores, sin `facts`, la plataforma lee el formato de
  `resumen_causas`.
- Un motivo nuevo de descarte debe registrar sus hechos al cerrar el trial; si no, su causa se
  cuenta sin ejemplo.
- Los números no finitos y los campos vacíos se omiten: `facts` siempre es JSON válido.

## Alternatives Considered

- **Interpretar `resumen_causas` en la plataforma.** Ata la interfaz al texto de un mensaje que
  el motor puede reescribir.
- **Interpretar los motivos en el motor al final de la búsqueda.** El motor volvería a leer su
  propio texto; los hechos ya existen como datos donde se decide el descarte.
