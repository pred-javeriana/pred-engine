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

### `rechazo.generar_series_aceptadas(serie_semilla, *, period, n_series=1, block_size=3, tolerancia=0.05, max_reintentos=20, semilla_aleatoria=42, generador=None, postproceso=None) -> ResultadoRechazo`

Por cada serie: genera candidatas hasta que una pase la divergencia o se agoten
los reintentos (`DivergenceRejectionExhausted`). `generador` es inyectable
(`Callable[[np.random.Generator], np.ndarray]`); por defecto STL +
`mbb.moving_block_bootstrap` sobre los residuales. `postproceso` se aplica a
cada candidata **antes** de evaluar la divergencia (la Fase 0 aplica alli las
leyes fisicas). Determinista para una `semilla_aleatoria` dada.
`ResultadoRechazo(series, semilla_aleatoria, intentos, rechazos, veredictos,
crudas)`; propiedad `tasa_rechazo = rechazos / intentos`.

### `rechazo.motor_mbb_directo(serie_semilla, *, block_size=30, ruido_relativo=0.03) -> GeneradorCandidata`

Motor de 0.2: MBB de la serie diaria completa (los bloques conservan las
rachas de cero), ruido gaussiano de `ruido_relativo` x media positiva solo en
los dias con demanda y piso de 1 unidad en esos dias. `rechazo.motor_mbb`
construye el motor STL + MBB de residuales de ADR-01-007.

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

`period=7, n_series_por_sku=10, block_size=None, tolerancia_divergencia=0.05,
max_reintentos=200, semilla_aleatoria=42, nombre_artefacto=...,
minimo_filas=50_000, incluir_semilla_en_panel=True, metodo="stl-mbb",
ruido_relativo=0.03, mapeo_columnas=()`. `block_size=None` usa el bloque del
metodo (`BLOQUE_POR_METODO`: 3 para `stl-mbb`, 30 para `mbb-directo`);
`bloque_efectivo` lo resuelve. Los parametros invalidos lanzan `ValueError`.

### `fase0.cuadricula_diaria(semilla) -> pd.DataFrame`

Calendario diario continuo por SKU: suma la demanda del mismo dia, rellena con
0 los dias sin registro y propaga el lead time vigente.

### `fase0.ejecutar_fase_0(ruta_semilla, config=None, *, data_root=None, reutilizar=False) -> ResultadoFase0`

Cadena One-Shot: carga la semilla (con `mapeo_columnas`) → calendario diario →
por SKU con historia suficiente genera `n_series_por_sku` replicas con el
metodo elegido y una semilla aleatoria propia por SKU → leyes fisicas antes de
la compuerta → concatena → `validar_conformidad_o_fallar` →
`exportar_artefacto_csv` (modo 0444) → `persistir_bitacora`. Con
`reutilizar=True`, si el artefacto ya existe y la bitacora mas reciente tiene la
misma huella de corrida y el mismo hash, devuelve ese resultado
(`ResultadoFase0.reutilizada=True`) en lugar de fallar por WORM.
`ResultadoFase0(artefacto, bitacora, bitacora_path, reutilizada)`. No importa
componentes del Modulo 1.

### `fase0.main(argv=None) -> int`

CLI `pred-engine-fase0 <semilla.csv> [--data-root ...] [--metodo stl-mbb|mbb-directo]
[--columna CANONICA=ORIGEN ...] [--period N] [--n-series N] [--block-size N]
[--ruido F] [--tolerancia F] [--max-reintentos N] [--seed N] [--nombre ...]
[--minimo-filas N] [--reutilizar]`. `agregar_opciones_fase0(parser, prefijo)`
registra las mismas opciones en otro CLI (`pred-engine run` usa `--m0-*`).
Errores esperables (`ERRORES_FASE0`): una linea `error: ...` y codigo 1.

### `bitacora.BitacoraCorrida` / `persistir_bitacora(bitacora, *, data_root=None) -> Path`

Dataclass serializable con los parametros reconstruibles de la corrida
(`semilla_aleatoria`, `tasa_rechazo`, `intentos_bootstrap`,
`n_demanda_rectificada`, `n_lead_time_acotado`, `row_count`, `artefacto_sha256`,
`iniciada_en`, `finalizada_en`, `metodo`, `block_size`, `ruido_relativo`,
`max_reintentos`, `mapeo_columnas`, `semilla_sha256`, `huella_corrida`,
`configuracion`). `leer_bitacoras(data_root=...)` devuelve las bitacoras de
`logs/` de la mas reciente a la mas antigua. `persistir_bitacora` escribe JSON en
`{data_root}/logs/fase0_<ts>_seed<N>.json` — fuera del directorio crudo.

## Errores (`pred_engine.aumentacion.errores`)

| Clase | Base | Cuando |
| --- | --- | --- |
| `PhysicalConstraintError` | `ValueError` | Violacion de una ley de conservacion fisica. |
| `DivergenceRejectionExhausted` | `RuntimeError` | Rejection sampling sin exito tras `max_reintentos`. |
| `SchemaConformanceError` | `ValueError` | El artefacto no cumple el contrato 0.4 (`.fallas`). |
| `WormOverwriteError` | `FileExistsError` | Segundo intento de escritura sobre `raw/`. |
