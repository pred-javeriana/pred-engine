# API — Fase 0 (secciones 0.3 y 0.4)

Identificadores en ingles para el contrato de datos; codigo y mensajes en
espaniol. Todo vive bajo `pred_engine.aumentacion`.

## 0.3 Compuerta de restricciones fisicas

### `restricciones.rectificar_demanda_no_negativa(demanda, *, umbral_inferior=0.0) -> ResultadoRectificacion`

Clipping asimetrico y **puro** (`np.maximum`). No muta la entrada.
`ResultadoRectificacion(valores: np.ndarray, n_rectificadas: int, porcentaje: float)`.

### `restricciones.truncar_a_unidades_enteras(demanda) -> np.ndarray[int64]`

`np.trunc(np.abs(x))`. Truncamiento, no redondeo.

### `restricciones.limites_lead_time_desde_semilla(semilla_lead_time) -> LimitesLeadTime`

`LimitesLeadTime(minimo: int, maximo: int)`, con `minimo >= 1` garantizado
(`floor` del minimo observado, `ceil` del maximo). Los limites invalidos lanzan
`PhysicalConstraintError`.

### `restricciones.acotar_lead_time(lead_time, limites) -> ResultadoAcotamiento`

`np.clip` al rango; `ResultadoAcotamiento(valores, n_acotadas, limites)`.
Emite `logger.warning` si `n_acotadas > 0`.

### `restricciones.aplicar_restricciones_fisicas(panel, *, limites_lead_time) -> ResultadoCompuerta`

Compone A1 + A2 + A3 sobre un `DataFrame` con las 4 columnas canonicas. No muta
el panel. `ResultadoCompuerta(panel, n_demanda_rectificada, n_lead_time_acotado,
limites_lead_time)`. Reeleva `PhysicalConstraintError` si algo escapa la ley.

### `divergencia.evaluar_divergencia_parametrica(semilla, candidata, *, tolerancia=0.05) -> VeredictoDivergencia`

Divergencia relativa `|x_cand - x_seed| / |x_seed|` de media y varianza globales.
`aceptada` sii ambas `<= tolerancia`. El veredicto incluye los seis estadisticos
calculados. `TOLERANCIA_DIVERGENCIA_POR_DEFECTO = 0.05`.

### `rechazo.generar_series_aceptadas(serie_semilla, *, period, n_series=1, block_size=3, tolerancia=0.05, max_reintentos=20, semilla_aleatoria=42, generador=None) -> ResultadoRechazo`

Por cada serie: genera candidatas hasta que una pase la divergencia o se agoten
los reintentos (`DivergenceRejectionExhausted`). `generador` es inyectable
(`Callable[[np.random.Generator], np.ndarray]`); por defecto STL +
`mbb.moving_block_bootstrap` sobre los residuales. Determinista para una
`semilla_aleatoria` dada.
`ResultadoRechazo(series: list[np.ndarray], semilla_aleatoria, intentos, rechazos,
veredictos)`; propiedad `tasa_rechazo = rechazos / intentos`.

## 0.4 Contrato de datos y persistencia

### `contrato`

- `CONTRACT_VERSION: str` — SemVer del esquema.
- `OUTPUT_COLUMNS = ("sku_id", "timestamp", "demand_qty", "lead_time_days")`.
- `OUTPUT_CONTRACT: tuple[ColumnSpec, ...]` — `ColumnSpec(name, dtype, description)`.
- `TIMESTAMP_FORMAT = "%Y-%m-%d"` (ISO 8601, fecha).
- `describir_contrato() -> str`.

### `conformidad.verificar_conformidad(frame) -> ReporteConformidad`

Nunca lanza. `ReporteConformidad(conforme: bool, contract_version, row_count,
verificaciones: tuple[Verificacion, ...])`; `.fallas` lista los checks fallidos.

### `conformidad.validar_conformidad_o_fallar(frame) -> ReporteConformidad`

Genera el reporte y lanza `SchemaConformanceError(mensaje, *, fallas)` con
`logger.critical` si `not reporte.conforme`.

### `worm.resolver_ruta_artefacto(nombre, *, data_root=None) -> Path`

`{data_root|PRED_DATA_ROOT|"data"}/raw/<nombre>`. `ValueError` si `nombre` es
ruta absoluta o contiene separadores.

### `worm.escribir_una_sola_vez(nombre, escritor, *, data_root=None) -> Path`

Ejecuta `escritor(destino: Path)` solo si el destino no existe; si existe lanza
`WormOverwriteError` (subclase de `FileExistsError`).

### `exportador_csv.exportar_artefacto_csv(frame, nombre="panel_sintetico_fase0.csv", *, data_root=None, minimo_filas=50_000) -> ArtefactoExportado`

Valida conformidad → exige `row_count >= minimo_filas` (`ValueError` si no) →
reordena a `OUTPUT_COLUMNS` y formatea `timestamp` a ISO → escribe via guarda
WORM → hashea. `ArtefactoExportado(path: Path, sha256: str, row_count: int)`.

## 0.4 Orquestacion

### `fase0.ConfiguracionCorrida`

`period=7, n_series_por_sku=10, block_size=3, tolerancia_divergencia=0.05,
max_reintentos=20, semilla_aleatoria=42, nombre_artefacto=...,
minimo_filas=50_000, incluir_semilla_en_panel=True`.

### `fase0.ejecutar_fase_0(ruta_semilla, config=None, *, data_root=None) -> ResultadoFase0`

Cadena One-Shot: carga semilla (exige las 4 columnas canonicas) → por SKU con
`len >= 2*period` genera `n_series_por_sku` replicas via `generar_series_aceptadas`
→ concatena → `aplicar_restricciones_fisicas` → `validar_conformidad_o_fallar` →
`exportar_artefacto_csv` → `persistir_bitacora`.
`ResultadoFase0(artefacto: ArtefactoExportado, bitacora: BitacoraCorrida,
bitacora_path: Path)`. No importa componentes del Modulo 1.

### `fase0.main(argv=None) -> int`

CLI: `python -m pred_engine.aumentacion.fase0 <semilla.csv> [--data-root ...]
[--period N] [--n-series N] [--block-size N] [--tolerancia F] [--max-reintentos N]
[--seed N] [--nombre ...] [--minimo-filas N]`.

### `bitacora.BitacoraCorrida` / `persistir_bitacora(bitacora, *, data_root=None) -> Path`

Dataclass serializable con los parametros reconstruibles de la corrida
(`semilla_aleatoria`, `tasa_rechazo`, `intentos_bootstrap`,
`n_demanda_rectificada`, `n_lead_time_acotado`, `row_count`, `artefacto_sha256`,
`iniciada_en`, `finalizada_en`). `persistir_bitacora` escribe JSON en
`{data_root}/logs/fase0_<ts>_seed<N>.json` — fuera del directorio crudo.

## Errores (`pred_engine.aumentacion.errores`)

| Clase | Base | Cuando |
| --- | --- | --- |
| `PhysicalConstraintError` | `ValueError` | Violacion de una ley de conservacion fisica. |
| `DivergenceRejectionExhausted` | `RuntimeError` | Rejection sampling sin exito tras `max_reintentos`. |
| `SchemaConformanceError` | `ValueError` | El artefacto no cumple el contrato 0.4 (`.fallas`). |
| `WormOverwriteError` | `FileExistsError` | Segundo intento de escritura sobre `raw/`. |
