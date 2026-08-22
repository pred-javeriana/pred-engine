# API — Ingesta 1.1

Superficie pública añadida en esta sesión. Los identificadores permanecen en
inglés (convención de la librería); el comportamiento se documenta aquí para
el jurado y para `pred-platform`.

## `pred_engine.comun.logger`

### `configure_json_logger(name="pred_engine", *, level=INFO, stream=None) -> Logger`

Configuración idempotente. El flujo por defecto es `sys.stdout`. Cada registro
es un objeto JSON con las claves `timestamp` (ISO-8601 UTC), `level`, `module`,
`file_hash`, `row_count`, `message`.

### `get_logger(name=None) -> Logger`

Garantiza que exista el manejador JSON de `pred_engine` y devuelve
`logging.getLogger(name)`.

### `log_ingestion_event(logger, message, *, file_hash, row_count, level=INFO) -> None`

Adjunta la telemetría obligatoria de ingesta mediante `extra=`.

### `JsonFormatter`

Subclase de `logging.Formatter`. Sin E/S.

## `pred_engine.ingesta.data`

### `ensure_data_layout(data_root=None) -> DataLayout`

Crea `{root}/raw`, `{root}/staging` y `{root}/processed` con `exist_ok=True`.
Raíz por defecto: variable de entorno `PRED_DATA_ROOT` o `Path("data")`.

### `DataLayout` (congelado)

Campos: `root`, `raw`, `staging`, `processed` (`pathlib.Path`).

### `raw_read_only_guard(raw_root)` / `enforce_raw_read_only(raw_root)`

Interceptor acotado de `builtins.open` e `io.open`. Los modos de escritura,
adición, creación exclusiva y actualización bajo `raw_root` lanzan
`RawWritePermissionError` (subclase de `PermissionError`) cuyo mensaje incluye
la ruta y la función llamante. La violación se registra en los logs.

## `pred_engine.ingesta.lector`

### `extract_csv(source, *, data_root=None) -> ExtractionArtifact`

Lee `.csv` desde `data_root/raw` con `dtype=str` (sin inferencia de fechas ni
coerción numérica). Devuelve `ExtractionArtifact` congelado
`(frame, sha256, row_count, source_path)`.

### `hash_sha256_archivo(ruta, tamano_bloque=65536) -> str`

SHA-256 por streaming sobre los bytes del archivo en disco.

### `export_parquet(frame, destination, *, data_root=None) -> Path`

Escribe Parquet comprimido con Snappy mediante `engine="pyarrow"` e
`index=False`. El destino debe estar bajo `data_root/processed`. Los destinos
bajo `raw/` lanzan `RawWritePermissionError`.
