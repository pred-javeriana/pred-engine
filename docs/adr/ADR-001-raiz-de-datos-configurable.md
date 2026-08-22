# ADR-001: Raíz de datos configurable

**Date:** 2026-08-22
**Status:** Proposed
**Notion Task:** TASK-DATA-1.1-B1 / TASK-DATA-1.1-B2

## Context

La arquitectura describe el árbol `/data/raw`, `/data/staging` y
`/data/processed` como contrato de almacenamiento inmutable. Interpretar esas
rutas como absolutas en el sistema de archivos rompería Windows, impediría
usar `tmp_path` en pytest y acoplaría la librería a un único despliegue.

## Decision

Todas las funciones de layout e I/O reciben `data_root: Path | None`.
La raíz por defecto es la variable de entorno `PRED_DATA_ROOT` o, si no
existe, `Path("data")` relativa al cwd del proceso. El contrato interno
(`raw/`, `staging/`, `processed/`) no cambia.

## Rationale

La librería no tiene capa de persistencia propia (README). Inyectar la raíz
permite reutilizar el 100 % del módulo con series de tiempo de otras
verticales y con el anfitrión `pred-platform`.

## Consequences

- Los scripts de ingesta y los tests deben pasar `data_root` o fijar
  `PRED_DATA_ROOT`.
- No se crean ni se protegen directorios en `/data` absoluto.
- Hay que sincronizar este ADR en Notion cuando se apruebe.

## Alternatives Considered

- **Rutas absolutas `/data/...`:** rechazado por portabilidad y testabilidad.
- **Constante de paquete apuntando al repo root:** rechazado porque la librería
  no debe asumir que se ejecuta desde el clon de git.
