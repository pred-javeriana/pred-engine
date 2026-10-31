# Evaluación 3.0: coordinación y recuperación

`ControladorEvaluacion` conecta el handoff real de M2 con el generador 3.3,
la evaluación 3.4 y la persistencia 3.5. Implementa el validador nativo de L4.
Recibe una conexión a la base única de plataforma; no crea otra base.

Antes de leer demanda reservada, valida contexto, clases, identidad y
configuraciones. Conserva el manifiesto emitido antes de la apertura y fija el
protocolo en `m3_protocolos`. Un lote inválido queda RECHAZADA con causa.

Cada candidato se reconstruye desde su configuración y semilla. Las unidades
confirmadas se cargan de SQLite. Solo se pronostican y confirman orígenes
pendientes. El ajuste se reconstruye si queda trabajo, pero no se repiten
pronósticos confirmados. La línea base usa el mismo calendario y sus propios
marcadores durables.

- EN_CURSO ejecuta unidades pendientes y confirma también las fallidas.
- EVALUADA calcula métricas desde las unidades durables y emite la selección.
- SELECCIONADA publica las exportaciones derivadas.
- PUBLICADA devuelve los veredictos existentes sin ejecutar modelos.
- RECHAZADA conserva el rechazo de frontera sin abrir ventanas.

El controlador no reemplaza la recuperación HPO de M2. Los estudios existentes
siguen conservando sus ensayos completos con su propio contrato 2.9. La
procedencia M3 registra las exclusiones y fallas de M2 sin inventar
configuraciones que M2 no entregó. Una unidad L2 fallida se repite al
reanudar, así que el reintento puede excluir o dejar de excluir un SKU con la
regla de dos familias (ADR-023). La receta aceptada no cambia: M3 compara los
candidatos sin esos SKU, juzga los SKU de su receta durable y publica las
exclusiones de su procedencia. La selección de robustez también queda en la
procedencia, fijada antes de abrir la reserva.

`tests/forecasting/test_control_evaluacion.py` prueba el flujo con M2 real,
caídas antes y después del commit, múltiples SKU, fallos aislados, rechazo del
handoff, reserva insuficiente y lectura inmutable de corridas publicadas.
