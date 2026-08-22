# ADR-002: Proveedores LLM desacoplados y alineacion fail-closed

**Date:** 2026-08-22
**Status:** Proposed
**Notion Task:** TASK-DATA-1.2-A1 / B1 / C1

## Context

La sonda de cabeceras debe hablar con un LLM a temperatura 0. El producto
final es un motor que la empresa instala, deposita CSVs e introduce *su*
clave. Soldar Gemini (u otro SDK) al modulo de ingesta romperia el desacople
y el criterio de reutilizacion 90 %. Ademas, un mapeo "aproximado" contaminaria
el modulo 2 de forecasting.

## Decision

1. Protocolo `LlmProvider` + fabrica `gemini | openai | anthropic` sobre `httpx`.
2. Catalogo explicito `AVAILABLE_MODELS` por proveedor (tier economico, ago 2026)
   con default al modelo mas barato (`DEFAULT_MODELS`). El CLI y la fabrica
   rechazan IDs fuera del catalogo (`UnknownModelError`).
3. Si el mapeo no cubre los cuatro campos canonicos, o el LLM inventa columnas,
   se lanza `SemanticAlignmentError` y se detiene la ingesta.
4. El contrato Pydantic (`InventoryObservation`) es el origen de verdad de
   los nombres canonico; la sonda no duplica la tupla a mano.
5. El dataset de proyecto ancla la semantica: demanda = uso (`Avg_Usage_Per_Day`),
   nunca stock on hand.

## Rationale

- Timeouts son criterio de aceptacion: `httpx` los trata como ciudadania de
  primera, sin tres SDKs y tres politicas de retry.
- Fail-closed es mas honesto ante el jurado y ante la empresa que un mapeo
  silencioso e incorrecto.
- Tres proveedores cubren las claves que un cliente corporativo suele tener.
- El catalogo de modelos hace transparente que tier se usa por defecto y que
  opciones puede elegir el operador (`pred-engine models`, `--model`).

## Consequences

- Hay que documentar `--provider`, `--model` y las variables de entorno de clave.
- Los tests de pytest mockean `complete()` o `httpx.Client`; la prueba de vida
  con Gemini es manual via CLI.
- Sincronizar este ADR en Notion cuando se apruebe.
- Actualizar `AVAILABLE_MODELS` cuando un proveedor retire o renombre modelos.

## Alternatives Considered

- **SDK oficial de un solo vendor:** rechazado (lock-in, timeouts heterogeneos).
- **LiteLLM:** rechazado (dependencia pesada, fuera de ISO 29110 proporcional).
- **Reglas regex de cabeceras:** rechazado por el propio Notion (cero hardcode
  de sinonimos); la sonda LLM es el mecanismo. El fail-closed evita que eso
  se convierta en alucinacion silenciosa.
- **Modelo libre sin catalogo:** rechazado; el operador debe elegir entre IDs
  conocidos y economicos, no strings arbitrarios que fallen en runtime opaco.
