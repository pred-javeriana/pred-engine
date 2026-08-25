## TASK-DATA-1.2-A1 — Construir probe de cabecera LLM diagnostico y generador de feedback

> **Modulo:** DATA | **Funcionalidad:** 1.2 Alineacion Semantica y Validacion | **Grupo:** A Alineacion Semantica

### La mision

- Como Marco de Ingesta de Datos,
- necesito implementar un asistente LLM diagnostico,
- para evaluar archivos fuente caoticos frente a nuestro esquema estricto y
  proporcionar a los usuarios feedback JSON accionable para corregir sus datos.

### Pasos

- Extraer una muestra representativa de las primeras 5 filas del DataFrame original.
- Formular un prompt zero-shot que instruya a un LLM, con `temperature=0.0`, a
  comparar las cabeceras de origen contra el contrato predefinido.
- Configurar el LLM para producir un payload JSON estricto con `status` y
  `diagnostic[]`.
- Implementar fail-fast: si `rejected`, registrar/imprimir JSON y detener el
  pipeline. **NO modificar el DataFrame.**

### Criterios de aceptacion

- [x] La integracion con el LLM es stateless y maneja timeouts.
- [x] Cero operaciones de mutacion (`df.rename`, `df.drop`).
- [x] Salida JSON estructurada con instrucciones de correccion.
- [x] Cobertura >= 80 % via `uv run pytest`.
- [x] Errores registrados via `logging` JSON.

## TASK-DATA-1.2-B1 — Implementar validacion estricta de esquemas con Pydantic

> **Modulo:** DATA | **Funcionalidad:** 1.2 | **Grupo:** B Validacion de Esquemas

Implementado en `pred_engine.ingesta.validador_formato` (`InventoryObservation`,
`validate_aligned_frame`). Requiere cabeceras canonicas exactas post-sonda.

## TASK-DATA-1.2-C1 — Desarrollar motor de remuestreo temporal sin estado

> **Modulo:** DATA | **Funcionalidad:** 1.2 | **Grupo:** C Continuidad Temporal

Implementado en `pred_engine.ingesta.continuidad` (`resample_daily`).
