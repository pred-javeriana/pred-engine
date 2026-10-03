# ADR-020: SARIMA admisible, elegibilidad del HPO en todas las ventanas y regla #3 sobre el pronóstico final

**Date:** 2026-10-02
**Status:** Accepted
**Notion ADR:** complementa ADR-02-005 (ASHA) y la regla #3 de poda semántica
de 2.4; la página de Notion no se modificó y queda pendiente registrar allí la
estimación restringida, el criterio de elegibilidad y la guarda del
pronóstico final.
**Notion Task:** 2.4 (selección clásica con HPO), 2.8 (walk-forward), 3.0
(recuperación por unidad)

## Context

ADR-018 dejó registrado un candidato clásico (`104::syn000`,
SARIMA(2,0,2)(2,0,0)7) con ventanas walk-forward explosivas (~1e61) que la
media recortada ocultaba. La causa no era ese candidato.

1. **Estimación sin restricción.** `SarimaForecaster` ajustaba con
   `enforce_stationarity=False` y `enforce_invertibility=False`. Sobre las
   ventanas de los 110 ganadores clásicos de la semilla Kaggle, 905 de 5500
   ajustes (16 %) quedaron con alguna raíz AR o MA dentro del círculo
   unitario, y 1237 no convergieron. En la ventana 12 de `104::syn000` el
   optimizador se detuvo con raíces de módulo cercano a 0 y una
   log-verosimilitud de −3456 (−711 con la estimación restringida); el
   pronóstico a 7 días llegó a 4,6e63.
2. **Borde de la región admisible.** Con la estimación restringida, el
   optimizador puede llevar una raíz hasta el borde: un SARIMA(2,2,2)(1,0,1)7
   terminó con raíces de módulo 1,00000 (AR) y 1,00007 (MA) y pronosticó
   220 898 unidades el primer día, con un máximo histórico de 486.
3. **Ventanas fallidas toleradas.** El HPO excluía del agregado las ventanas
   sin ajuste y daba por completo el trial. Con la guarda del punto anterior,
   ganaban configuraciones evaluadas en 4 de 49 ventanas que luego no
   ajustaban con la historia completa (53 unidades fallidas de 208).
4. **Pronóstico final sin la regla #3.** L2 solo exigía que el pronóstico
   desde t* fuera finito. El candidato `100::syn002`, SARIMA(2,2,2)(1,0,1)7,
   entregó a M3 cero en los 100 días reservados: su tendencia negativa quedaba
   recortada a cero.

## Decision

1. `SarimaForecaster` estima con `enforce_stationarity=True` y
   `enforce_invertibility=True`: la parte ARMA del modelo diferenciado es
   estacionaria e invertible (Box y Jenkins).
2. Un ajuste con alguna raíz AR o MA de módulo menor que 1,01 se rechaza con
   `AjusteModeloError`, el mismo umbral que usa `auto.arima` del paquete
   forecast (Hyndman y Khandakar, 2008) para descartar modelos inestables. Un
   ajuste fallido invalida el modelo anterior.
3. En el HPO, una ventana sin ajuste cierra el trial como `fallido`
   (`ajuste_fallido`). Todos los trials se comparan en las mismas ventanas, y
   L3 vuelve a ajustarlas sin tolerar fallos.
4. L2 aplica al pronóstico desde t* la misma regla #3 que el HPO aplica a cada
   ventana (no finito, negativo o nulo con historia positiva). Un pronóstico
   degenerado hace fallar su unidad con la causa; no se entrega a M3.

## Rationale

El pronóstico explosivo era una falla del optimizador fuera del espacio
admisible del modelo, no una propiedad de la configuración. Restringir la
estimación lo elimina en el origen, y el umbral de las raíces descarta los
ajustes que quedan en el borde, donde el pronóstico pierde estabilidad
numérica. Ninguna de las dos guardas recorta valores: rechazan el ajuste. Una
configuración que no se puede estimar en alguna ventana no se compara en
igualdad con las demás, ni resiste el reajuste de L3 o de M3. La regla #3 es
la definición del proyecto de un pronóstico degenerado. Aplicarla también al
pronóstico final asegura que M3 solo reciba pronósticos que habrían pasado el
HPO.

## Consequences

- Corrida de referencia con la semilla Kaggle: ninguna ventana walk-forward
  supera 1,12 veces el máximo histórico de su SKU hasta t*, ningún pronóstico
  final lo supera (máximo 0,56 veces) y ninguno es nulo. En 80 de los 105
  candidatos clásicos el HPO elige `d = D = 0` con constante.
- La guarda reduce el conjunto de configuraciones elegibles. Una unidad clásica
  falla de forma aislada cuando ningún trial ajusta en todas las ventanas o
  cuando el pronóstico final del ganador es degenerado (código de salida 8).
  La guía de ejecución registra cuántas fallan en la corrida de referencia.
- La política 2.2 enruta los SKU `lumpy` solo a la familia clásica; si esa
  unidad falla, el SKU queda sin candidato para M3. Elegir el siguiente trial
  elegible cuando el ganador falla en L2 queda como decisión pendiente.
- Los trials fallidos se cierran en la primera ventana sin ajuste, así que el
  HPO clásico termina antes.

## Alternatives Considered

- **Umbral de explosión (k veces el máximo histórico):** rechazado. La
  constante sería arbitraria y no explica la causa.
- **Recortar el pronóstico:** rechazado. Oculta el defecto del ajuste.
- **Mantener la tolerancia de ventanas del HPO y tolerar también en L3:**
  rechazado. Medido: ganan configuraciones evaluadas en una fracción de las
  ventanas y 53 de 208 unidades fallan al ajustarse con la historia completa.

## Related

- ADR-013 (ASHA), ADR-017 (ejecución de M2), ADR-018 (clásica en demanda
  intermitente).
- `src/pred_engine/comun/modelos/modelos_clasicos/sarima.py`,
  `src/pred_engine/optimizacion/optimizadores/HPO/estudio.py`,
  `src/pred_engine/pipeline.py`.
