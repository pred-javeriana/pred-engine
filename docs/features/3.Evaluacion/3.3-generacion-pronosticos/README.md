# Evaluación 3.3: pronósticos fechados sobre la reserva

`forecasting.generador_pronosticos.generar_pronosticos` ajusta una configuración
validada de M2 con la historia hasta t* y genera cada ventana completa del
periodo reservado. No ejecuta HPO ni modifica la configuración recibida.

## Protocolo causal

- La reserva sigue el corte común de `comun.reserva.ReserveCut`, con el 20 %
  final del calendario. El horizonte H se fija antes de abrirla.
- Los orígenes son diarios desde t* hasta `last_observed - H días`, ambos
  incluidos. El paso usado por el HPO no interviene.
- Cada ventana tiene exactamente H días. El generador no acorta la última.
- El modelo se ajusta una vez con la historia hasta t*. En cada origen recibe
  únicamente las observaciones reservadas disponibles hasta ese origen, sin
  reestimar parámetros.
- Cambiar datos posteriores a un origen no cambia el pronóstico de ese origen.
  Cambiar una observación ya disponible sí puede cambiar orígenes posteriores.

Esta interpretación aplica la recomendación de ADR-03-004. El texto antiguo
ADR-03-001 describe reajustes expansivos y TASK-EVAL-3.3-A2 formula invariancia
respecto de toda la reserva. No son equivalentes: la garantía aplicada aquí es
causalidad respecto de cada origen y ajuste único en t*.

## Fallos y exclusiones

Una historia insuficiente, inválida o con huecos se excluye antes del ajuste,
con SKU y código explícitos. Una reserva menor que H también se excluye sin
cambiar H. La historia mínima se declara en la configuración y no puede ser
menor que los siete días que necesita la línea base.

Una excepción de ajuste, excepción de pronóstico, salida no finita o dimensión
incorrecta produce un `FalloPronostico`. La unidad conserva las fechas
planeadas con NaN, nunca ceros imputados. Los demás orígenes continúan. Una
observación ausente antes del origen invalida ese origen y los siguientes que
la necesiten. Un error grande pero finito se conserva como evidencia, sin
aplicar la poda semántica de M2.

## Recuperación por unidad

El coordinador puede pasar las unidades durables en `confirmadas` y una
función `confirmar` que guarde cada unidad nueva con su marcador atómico.
El generador no recalcula ni vuelve a confirmar las unidades recibidas.
Si todas están confirmadas, tampoco instancia ni ajusta el modelo.

Si queda trabajo pendiente, reconstruye el ajuste con la historia permitida
y ejecuta solo esos orígenes. Las unidades incompatibles con candidato,
versión, fechas o horizonte se rechazan. Los errores de persistencia se
propagan: nunca se convierten en fallos técnicos del modelo.

## Integración

La salida expone `SerieCandidato` para 3.4, más la trazabilidad de configuración,
semilla, versión de código, emisión y fallos por origen. El generador no abre
archivos, no crea tablas y no decide cuándo se abre la reserva. El coordinador
debe validar primero el handoff mediante 3.2 y conservar el protocolo de la
corrida antes de invocarlo.

## Verificación

`tests/forecasting/test_generacion_reserva.py` cubre ventanas completas,
causalidad por origen, exclusiones, aislamiento, valores extremos válidos,
protección de la entrada y reinicios antes y después de confirmar unidades.
Las pruebas de los adaptadores en `tests/forecasting/adaptador_candidatos/`
comprueban la implementación concreta de `pronosticar` por familia.

3.3 pronostica toda la reserva. M3 elige la familia campeona en el bloque de
selección y puntúa al campeón en el bloque de prueba (ADR-023). Esa evidencia
sigue siendo retrospectiva sobre un solo corte t*.
