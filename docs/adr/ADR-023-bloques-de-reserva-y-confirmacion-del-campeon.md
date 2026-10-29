# ADR-023: Bloques de selección y de prueba en la reserva, confirmación del campeón y regla de dos familias

**Fecha:** 2026-10-29
**Estado:** Aceptada
**Referencias:** ADR-03-003, ADR-03-007, ADR-03-008 y ADR-03-011 del Hub;
ADR-020, ADR-021 y ADR-022; tareas 3.0, 3.4 y 3.5.
**Reemplaza en parte:** ADR-021, decisión 3 (Benjamini-Hochberg entre todos
los SKU de la corrida) y sección «Límites» (la familia campeona se elegía sobre
la misma reserva que producía su puntaje).

## Contexto

M3 elegía la familia campeona de cada categoría con la mediana de r en toda la
reserva y puntuaba al campeón en esa misma reserva, así que su puntaje dependía
de su propia selección. Benjamini-Hochberg cubría todos los SKU de la corrida:
la categoría más grande fijaba el umbral de las pequeñas. Ninguna prueba
comparaba al campeón con las demás familias de su categoría. Un SKU con una
sola familia evaluable tras L2 llegaba a M3 y solo competía contra Seasonal
Naive, con un ganador obvio.

## Decisiones

1. **Dos bloques de la reserva.** Con R días reservados, horizonte H y fracción
   f = 0,4, los primeros S = floor(f · R) días forman el bloque de selección y
   los demás el bloque de prueba. El origen t* + k pronostica los días k + 1 a
   k + H. Los orígenes k = 0 a S − H son de selección. Los orígenes k = S − H + 1
   a S − 1 cruzan el límite y forman un embargo que nada usa. Los orígenes
   k = S a R − H son de prueba. La fracción pertenece a la política de selección
   `3.4.4-bloques`. Los límites se fijan con el protocolo antes de abrir la
   reserva, en `m3_protocolos.protocolo`, clave `bloques`. Cada candidato se
   sigue ajustando una sola vez en t* (ADR-021, decisión 1).
2. **Campeón congelado en el bloque de selección.** La regla de la categoría
   (familias elegibles, representante de menor RMSE, conjunto común, mediana de
   r, compuerta y desempate) solo lee el bloque de selección. Un SKU entra en el
   conjunto común solo si cada familia que compite tiene al menos `n_min`
   ventanas de selección comunes con Seasonal Naive. La familia campeona y la
   instancia de cada familia en cada SKU quedan congeladas: el bloque de prueba
   nunca las cambia.
3. **Veredictos en el bloque de prueba.** Los cinco veredictos conservan sus
   nombres y su orden de reglas (ADR-03-008), con cada condición leída en el
   bloque de prueba. La significancia va en su propia columna y nunca cambia el
   veredicto. Un SKU sin ninguna familia evaluable tras L2 conserva
   `FALLO_TECNICO`.
4. **Significancia por categoría.** HLN-DM del campeón frente a Seasonal Naive
   en el bloque de prueba, con Benjamini-Hochberg dentro de cada categoría al
   nivel `alfa_dm` (0,05). Los conteos con Benjamini-Yekutieli se informan como
   sensibilidad. Cada SKU probado lleva el intervalo de r que invierte la prueba
   (Fieller) al nivel 1 − `alfa_dm`. Un SKU que BH selecciona lleva además el
   mismo intervalo al nivel 1 − R · q / m de su categoría, con R rechazos entre
   m pruebas (Benjamini y Yekutieli, 2005). Cuando la pérdida de Seasonal Naive
   no se distingue de cero, la inversión puede dar dos tramos,
   [inferior, superior] y [inferior_cola, ∞), y la publicación guarda los dos.
5. **Confirmación del campeón.** En el bloque de prueba, el campeón se compara
   con cada competidora de su categoría en los SKU que tienen ambas instancias
   y al menos `n_min_dm` ventanas comunes. Para cada SKU,
   x = log(RMSE_campeón / RMSE_competidora). La prueba de Wilcoxon de rangos con
   signo, bilateral, cuenta solo si la mediana de x es negativa; si no, su p
   vale 1. El p de la categoría es el mayor entre sus competidoras
   (intersección-unión) y Holm lo ajusta entre categorías. El campeón queda
   `confirmado` si el p de Holm es <= `alfa_dm` y, para cada competidora, el
   intervalo bootstrap del 95 % de la mediana de razones queda bajo 1. El
   bootstrap remuestrea bloques circulares de 14 días de prueba, con 2000
   réplicas y semilla fija, y usa los mismos días para todos los SKU. Con menos
   de 10 SKU para alguna competidora, la categoría queda `sin_comparables`. Si
   el campeón es Seasonal Naive o los datos no son reales declarados, queda
   `no_aplica`. Una confirmación fallida queda `no_confirmado` y nunca cambia
   al campeón. Cada comparación informa también Hodges-Lehmann y la proporción
   de victorias con intervalo de Wilson. Un SKU con pérdida cero también
   compite, sin epsilon: su razón vale 0 si solo el campeón no tiene error,
   infinito si solo la competidora no lo tiene y 1 si ninguno lo tiene. Un
   valor que combina una razón 0 y una infinita queda sin definir y la causa
   `perdida_cero:<campos>` lo nombra.
6. **Contendiente.** La misma regla de selección, sin la familia fundacional,
   elige la mejor familia que no es Chronos-2. Su fila de comparación mide a
   Chronos-2 frente a ella cuando Chronos-2 es el campeón.
7. **Selección de robustez.** La misma regla de selección se aplica a los
   pliegues L3, que solo usan historia hasta t*, con Seasonal Naive ajustado en
   cada pliegue. Se fija con el protocolo y solo se informa. Nunca elige al
   campeón: las familias optimizadas se puntúan en los pliegues que su HPO usó.
8. **Regla de dos familias.** Una familia es evaluable para un SKU si su
   unidad L2 terminó y entregó un candidato. Al cerrar L2, un SKU con una sola
   familia evaluable sale de la competencia con
   `eligibility.fewer_than_two_families` (ADR-022), sea cual sea el número de
   familias enrutadas. No llega a L3 ni recibe veredicto. Sin ninguna familia
   evaluable, el SKU sigue a M3, que lo juzga `FALLO_TECNICO`. Por eso una
   corrida de una sola familia, o un SKU `lumpy` en una corrida sin la familia
   fundacional, no llega a M3. Una unidad L2 fallida no queda en checkpoint y se
   repite al reanudar, así que un reintento puede cambiar la exclusión. Por eso
   M3 honra las exclusiones de la procedencia durable de su protocolo.
9. **Publicación.** La migración 0011 (`esquema_m3_0007.sql`) agrega
   `m3_significancia`, `m3_confirmaciones`, `m3_comparaciones`,
   `m3_seleccion_previa`, `m3_intervalos_r` y `m3_exclusiones`. Las tablas
   anteriores conservan sus columnas. `m3_veredictos`, `m3_cartera` y las filas
   `agregada` de `m3_metricas` leen el bloque de prueba. Las filas por ventana de
   `m3_metricas` cubren toda la reserva. `m3_exclusiones` lista los SKU que M2
   excluyó, con su categoría, código y causa.
10. **Grupo final de corridas.** La evidencia final es un grupo de tres
    corridas que solo difieren en la semilla del motor: 0, 1 y 2. Cada corrida
    publica su significancia propia (decisión 4), que no es la del grupo,
    porque las corridas usan los mismos productos y días. Quien agrupa las
    corridas aplica esta regla y M3 sigue publicando por corrida. Las entradas
    son `m3_veredictos.dm_p_valor` (p crudo de HLN-DM) y el signo de
    `m3_veredictos.dm_estadistico`, que es el signo del diferencial campeón
    menos Seasonal Naive cuando la prueba es válida. Un producto entra en la
    familia de prueba del grupo solo si su `dm_p_valor` no es nulo en las tres
    corridas. Su p del grupo es el mayor de esos tres p, un p válido con
    cualquier dependencia (intersección-unión). Benjamini-Hochberg se aplica una
    sola vez dentro de cada categoría, a q = 0,05, sobre los p del grupo. Un
    producto es significativamente mejor que Seasonal Naive solo si BH lo
    rechaza y su estadístico es negativo en las tres corridas. Un producto con
    signos mixtos sigue en la familia de BH y nunca cuenta como mejor. Los
    descubrimientos de corridas distintas nunca se suman, los p nunca se
    promedian ni se combinan con Fisher o Stouffer, y un producto sin p en una
    corrida nunca entra con el mayor de sus p disponibles. La guía 3.4 tiene un
    ejemplo trabajado que la verificación del agrupador debe reproducir.

## Consecuencias

- El bloque de prueba es más corto que la reserva. Un SKU necesita `n_min`
  ventanas válidas de prueba para pasar de `EVIDENCIA_INSUFICIENTE`, y el
  conjunto común necesita `n_min` ventanas de selección por familia.
- Un periodo del año que solo cae en el bloque de selección no tiene resultado
  en la prueba. Se describe con las métricas por ventana de la reserva.
- La regla de dos familias no elige SKU al azar: los resultados describen los
  SKU en que al menos dos familias producen un modelo válido en M2. La
  publicación cuenta las exclusiones por categoría y código.
- pred-platform aplica el texto de `esquema_m3_0007.sql` como su migración 0011
  antes de leer corridas con estas tablas.

## Alternativas consideradas

- **Seleccionar en los pliegues L3 y probar en toda la reserva.** Las familias
  optimizadas se puntúan en los pliegues de su HPO y su error L3 es optimista,
  mientras Chronos-2 no se optimiza. Queda como selección de robustez.
- **Tres bloques, con un HPO que termina antes de un periodo de validación.**
  Separa todo, pero exige otro ajuste de M2 con una historia más corta.
- **Benjamini-Hochberg entre todos los SKU de la corrida.** La categoría más
  grande fija el umbral de las pequeñas, y las afirmaciones son por categoría.
- **`VALIDADO` condicionado a la significancia.** Contradice ADR-03-008, que
  separa el veredicto descriptivo de la evidencia estadística.
- **Model Confidence Set o Diebold-Mariano de panel.** El primero exige otro
  bootstrap y aporta poco con dos a cuatro familias. En el segundo, pocos SKU
  con pérdida de Seasonal Naive pequeña dominan el promedio.

## Límites

La evidencia sigue siendo retrospectiva sobre un solo corte t*. Wilcoxon trata
los SKU como independientes; el bootstrap de días comunes cubre la dependencia
entre SKU y la serial hasta unas dos semanas. «Sin diferencia significativa»
nunca significa equivalencia.
