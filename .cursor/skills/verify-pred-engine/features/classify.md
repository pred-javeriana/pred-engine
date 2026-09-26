# SKU topology classification (Syntetos-Boylan)

SKU topology classification calculates the Average Demand Interval (ADI) and squared Coefficient of Variation (CV²) for each SKU series to classify demand into one of four Syntetos-Boylan quadrants (`smooth`, `intermittent`, `erratic`, `lumpy`), verifies transactional column preservation, and outputs a validated Snappy Parquet artifact to `<data_root>/processed/`.

## Sub-features

- `classify-csv` loads a 4-column canonical CSV, validates column schema, performs daily grid resampling (filling missing dates with demand 0), calculates topology, and publishes Parquet.
- `classify-parquet` processes an existing daily panel Parquet file (from Module 1.2), validates schema, skips redundant resampling, calculates topology, and publishes Parquet.
- `classify-syntetos-boylan` applies fixed classification thresholds (ADI cutoff 1.32, CV² cutoff 0.49) to tag each row with its `sku_class`.
- `classify-contract` guarantees that no input demand or timestamp values are altered, and that exactly one new column (`sku_class`) is added to the 1.4 output schema.

## How to get to it (user POV)

- Run `uv run pred-engine classify --csv <csv_path> --data-root <data_root>`.
- Run `uv run pred-engine classify --parquet <parquet_path> --data-root <data_root>`.
- Run through harness: `./.cursor/skills/verify-pred-engine/harness.sh drive-classify <evidence_dir> <csv_path> [data_root]`.

## Driving it with harness.sh

Preconditions:

- Runtime environment passes `./.cursor/skills/verify-pred-engine/harness.sh doctor`.
- Input file contains canonical schema columns: `sku_id`, `timestamp`, `demand_qty`, and `lead_time_days`.
- Each SKU contains at least one observation with positive demand (`demand_qty > 0`).

- **Classify from CSV.** Run `./.cursor/skills/verify-pred-engine/harness.sh drive-classify <evidence_dir> <csv_path> <data_root>` or run directly:
  ```bash
  uv run pred-engine classify --csv /tmp/inventory.csv --data-root /tmp/data
  ```
  Exit code is `0`. Stdout outputs:
  - Table of computed SKU metrics: `sku_id,n_periods,n_positive,adi,cv2,sku_class`.
  - Distribution dictionary: `sku_class_resumen: {'intermittent': N, ...}`.
  - Contract verdict: `contrato_1_4: accepted`.
  - Destination artifact confirmation: `parquet: <data_root>/processed/inventory.parquet`.
- **Confirm Parquet file generation.** Check filesystem for `<data_root>/processed/<stem>.parquet`. Verify file is a valid non-empty Snappy Parquet file.
- **Confirm contract acceptance.** Parse stdout to confirm `contrato_1_4: accepted` appears and that `columnas:` lists exactly `sku_id,timestamp,demand_qty,lead_time_days,sku_class`.
- **Verify zero-demand rejection.** Run classify against a CSV where a SKU has all zero demand. Exit code is `6` (`OutputHandoffError`) with stderr indicating that demand cannot be entirely zero for topology calculations.
- **Proof capture.** Check `<evidence_dir>/classify/status.txt`, `<evidence_dir>/classify/stdout.txt`, and file size metadata.

## Gotchas

- You must pass exactly one input source: either `--csv` or `--parquet`. Passing neither or both produces exit code `1` with an error message `Pase exactamente uno de --csv o --parquet`.
- SKUs without any positive demand cannot compute ADI (division by zero) and will trigger an `OutputHandoffError` with exit code `6`.
- The classification module does not touch an LLM provider and requires no API key.
- The destination file is always saved in `<data_root>/processed/` with the `.parquet` extension, even if the source was a `.csv`.
