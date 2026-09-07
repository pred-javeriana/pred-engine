## TASK-DATA-1.4-A1 — Implementar validador del contrato final

Implementado en `pred_engine.ingesta.salida.validador` y
`ClassifiedObservation`.

### Criterios de aceptacion

- [x] El validador rechaza cualquier DataFrame que no contenga exactamente el contrato de cinco columnas.
- [x] `sku_id` cumple el tipo String definido por el contrato.
- [x] `timestamp` cumple el tipo Datetime64 y no contiene valores nulos.
- [x] `demand_qty` cumple el tipo Float, no contiene valores nulos y todos sus valores son >= 0.
- [x] `lead_time_days` cumple el tipo Integer y respeta las restricciones heredadas de 1.2.
- [x] `sku_class` solo admite `smooth`, `intermittent`, `erratic` y `lumpy`.
- [x] Cada SKU posee exactamente una clasificacion constante en todas sus filas.
- [x] La validacion opera de forma fail-closed y no modifica el DataFrame recibido.
- [x] La funcionalidad funciona segun lo definido.
- [x] El codigo es modular y demuestra un 90 % de reutilizacion para verticales alternativas.
- [x] `README.md` y `API_SPECIFICATION.md` reflejan el estado actual.
- [x] Los ADR-01-009, ADR-01-010, ADR-01-011 y ADR-01-012 reflejan el estado final de la implementacion.
- [x] La cobertura de pruebas se verifica en >= 80 % mediante `make test-cov`.
- [x] Los errores se registran mediante el modulo `logging` de Python (`logger`).
- [ ] El PR es aprobado por al menos otro integrante y se integra mediante _squash merge_.
- [ ] La tarea se mueve a `Completed`.
- [x] No se registra nueva informacion de identificacion personal (PII/PHI) en los registros.
- [x] La ejecucion pasa `ruff check` sobre el codigo de 1.4 y `ruff format`.
- [x] El tipado estricto es validado por `pyright` sin errores.
- [ ] Los _pre-commit hooks_ estan instalados y verificados en el entorno local.

## TASK-DATA-1.4-A2 — Verificar preservacion del panel clasificado

Implementado en `pred_engine.ingesta.salida.preservacion`. Se ejecuta
dentro de `publish_classified_panel` antes de escribir.

## TASK-DATA-1.4-B1 — Publicar artefacto final en Parquet

Implementado en `pred_engine.ingesta.salida.publicador`.
Ruta `{data_root}/processed/<stem>.parquet`.

## TASK-DATA-1.4-C1 — Validar handoff completo de ingesta

Implementado en `run_classify_csv` / `run_ingest` + `run_verify_parquet`.
Prueba de integracion: `tests/ingesta/salida/test_handoff.py`.
Dataset oficial: 4862 filas, 10 SKU, todos `intermittent`.
