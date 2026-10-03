# ADR-019: M1 clasifica cada SKU sin los días reservados para M3

**Date:** 2026-10-02
**Status:** Accepted
**Notion ADR:** aplica ADR-03-003 (reserva cronológica del 20 %) a la
clasificación de 1.3; la página de Notion no se modificó y queda pendiente
registrar allí que la clase se calcula hasta t*.
**Notion Task:** 1.3 (motor de topología), 1.4 (artefacto de salida), 3.1
(reserva temporal y causalidad)

## Context

M2 solo usa la historia hasta t* (ADR-017), pero la clase Syntetos-Boylan que
decide la ruta de familias llegaba de M1, que calculaba ADI y CV² sobre el
panel completo, incluidos los días reservados. La demanda de la reserva podía
cambiar la clase y, con ella, las familias que M2 optimiza. La misma clase
viaja a M3 en el manifiesto de candidatos. ADR-017 dejó esta fuga como
decisión pendiente.

## Decision

1. El corte de ADR-03-003 (`ReserveCut`, `RESERVE_FRACTION`) vive en
   `comun.reserva`. M1 y M2 lo derivan del calendario del panel con la misma
   función, así que coinciden sin intercambiar estado.
2. `classify_panel` calcula ADI y CV² de cada SKU solo con las filas hasta
   t*. El panel publicado conserva todas las filas: M3 necesita la reserva.
3. La compuerta de demanda positiva previa a la clasificación revisa la misma
   historia. Un SKU sin demanda positiva antes de t* se rechaza con su causa,
   y un panel demasiado corto para dejar historia admisible también.
4. `TopologyMetrics` cuenta los periodos y las demandas de la historia
   admisible.

## Rationale

Ninguna decisión de M2 debe depender de datos posteriores a t*. Calcular la
clase en M1 con la historia admisible evita la fuga en su origen: el artefacto
1.4, el enrutador y el manifiesto que recibe M3 llevan la misma clase, sin
fuga.

## Consequences

- Con la semilla Kaggle, M1 clasifica 98 SKU `intermittent` y 12 `lumpy`
  (antes, con la reserva incluida, 102 y 8).
- `pred-engine classify` también aplica el corte: todo panel que publica M1
  queda listo para M2 sin reclasificar.
- Un panel de un solo día ya no se puede clasificar.

## Alternatives Considered

- **Reclasificar solo en M2:** rechazado. El artefacto 1.4 seguiría llevando
  una clase calculada con la reserva, y esa clase llega a M3.
- **Pasar t* a M1 como parámetro:** rechazado. t* ya queda definido por el
  calendario del panel y la fracción fija de ADR-03-003.

## Related

- ADR-003 (motor de topología), ADR-017 (ejecución de M2 con reserva).
- `src/pred_engine/comun/reserva.py`,
  `src/pred_engine/ingesta/categorizacion/panel.py`,
  `src/pred_engine/ingesta/salida/precondiciones.py`.
