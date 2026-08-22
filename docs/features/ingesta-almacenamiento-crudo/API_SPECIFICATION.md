# API — Ingesta 1.1

Public surface added in this session. Identifiers are English (library
convention); behaviour is documented here for the jury and for `pred-platform`.

## `pred_engine.comun.logger`

### `configure_json_logger(name="pred_engine", *, level=INFO, stream=None) -> Logger`

Idempotent setup. Default stream is `sys.stdout`. Each record is one JSON
object with keys `timestamp` (ISO-8601 UTC), `level`, `module`, `file_hash`,
`row_count`, `message`.

### `get_logger(name=None) -> Logger`

Ensures the `pred_engine` JSON handler exists, then returns `logging.getLogger(name)`.

### `log_ingestion_event(logger, message, *, file_hash, row_count, level=INFO) -> None`

Attaches mandatory ingestion telemetry via `extra=`.

### `JsonFormatter`

`logging.Formatter` subclass. No I/O.

## `pred_engine.ingesta.data`

### `ensure_data_layout(data_root=None) -> DataLayout`

Creates `{root}/raw`, `{root}/staging`, `{root}/processed` with `exist_ok=True`.
Default root: env `PRED_DATA_ROOT` or `Path("data")`.

### `DataLayout` (frozen)

Fields: `root`, `raw`, `staging`, `processed` (`pathlib.Path`).

### `raw_read_only_guard(raw_root)` / `enforce_raw_read_only(raw_root)`

Scoped interceptor of `builtins.open`. Write/append/exclusive/update modes
under `raw_root` raise `RawWritePermissionError` (a `PermissionError`) whose
message includes the path and the calling function. The violation is logged.

## `pred_engine.ingesta.lector`

### `extract_csv(source, *, data_root=None) -> ExtractionArtifact`

Reads `.csv` from `data_root/raw` with `dtype=str` (no date inference, no
numeric coercion). Returns frozen `ExtractionArtifact(frame, sha256, row_count, source_path)`.

### `hash_sha256_archivo(ruta, tamano_bloque=65536) -> str`

Streaming SHA-256 of on-disk bytes.

### `export_parquet(frame, destination, *, data_root=None) -> Path`

Writes Snappy Parquet via `engine="pyarrow"` and `index=False`. Destination
must be under `data_root/processed`. Destinations under `raw/` raise
`RawWritePermissionError`.
