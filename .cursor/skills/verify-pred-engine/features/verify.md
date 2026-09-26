# Output Parquet contract verification

Output Parquet verification re-reads a published 1.4 Snappy Parquet artifact and independently audits schema conformity, required columns, absence of nulls, non-negative demand, bounded lead time, and valid Syntetos-Boylan class labels.

## Sub-features

- `verify-valid` verifies that a 5-column Snappy Parquet file satisfies all 1.4 data invariants and outputs `contrato_1_4: accepted`.
- `verify-schema-audit` checks that the five mandatory columns (`sku_id`, `timestamp`, `demand_qty`, `lead_time_days`, `sku_class`) are present with expected data types.
- `verify-bounds` confirms that `demand_qty` values are non-negative and `lead_time_days` values are greater than or equal to 1.
- `verify-classes` validates that all values in the `sku_class` column belong to the set `{"smooth", "intermittent", "erratic", "lumpy"}`.

## How to get to it (user POV)

- Run `uv run pred-engine verify --parquet <parquet_path>`.
- Run through harness: `./.cursor/skills/verify-pred-engine/harness.sh drive-verify <evidence_dir> <parquet_path>`.

## Driving it with harness.sh

Preconditions:

- Runtime environment passes `./.cursor/skills/verify-pred-engine/harness.sh doctor`.
- A candidate Snappy Parquet file exists (e.g. generated via `pred-engine classify` or `pred-engine ingest`).

- **Verify conforming Parquet artifact.** Run `./.cursor/skills/verify-pred-engine/harness.sh drive-verify <evidence_dir> <parquet_path>` or run directly:
  ```bash
  uv run pred-engine verify --parquet /tmp/data/processed/inventory.parquet
  ```
  Exit code is `0`. Stdout outputs:
  - Acceptance line: `contrato_1_4: accepted`.
  - Absolute resolved path: `parquet: <path>`.
  - Row and SKU summary: `filas: <N>`, `skus: <M>`.
  - Column list: `columnas: sku_id,timestamp,demand_qty,lead_time_days,sku_class`.
  - Per-SKU topology summary: `sku_class_resumen_sku: {'intermittent': M, ...}`.
- **Reject missing file.** Run `uv run pred-engine verify --parquet /tmp/missing.parquet`. Exit code is `1` and stderr reports `FileNotFoundError`.
- **Reject non-conforming schema.** Run verify on a Parquet missing `sku_class` or with corrupted columns. Exit code is `6` (`OutputHandoffError`).
- **Proof capture.** Check `<evidence_dir>/verify/status.txt` and `<evidence_dir>/verify/stdout.txt` for the `contrato_1_4: accepted` confirmation marker.

## Gotchas

- The `verify` subcommand strictly expects a Parquet file; passing a CSV path will trigger a PyArrow Parquet deserialization failure.
- Contract violation triggers exit code `6` (`OutputHandoffError`), whereas file-not-found or argument errors produce exit code `1`.
- The command performs a full audit scan of the entire dataset; large multi-million row Parquet files will be scanned sequentially.
- No LLM provider or API key is required.
