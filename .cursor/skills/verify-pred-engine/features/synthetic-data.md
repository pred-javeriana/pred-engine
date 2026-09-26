# Pre-ingestion synthetic data simulation (Phase 0)

Pre-ingestion synthetic data simulation runs a one-shot data augmentation orchestrator (`pred-engine-fase0`) that bootstraps historical seed series via STL decomposition and Moving Block Bootstrap (MBB), enforces physical supply chain constraints, validates schema conformity, and publishes a Write-Once-Read-Many (WORM) CSV and JSON run audit log.

## Sub-features

- `fase0-orchestrate` executes end-to-end bootstrap resampling, rejection sampling, clipping, conformity checking, and export in a single command.
- `fase0-worm` sets strict read-only permissions on the generated raw CSV under `<data_root>/raw/panel_sintetico_fase0.csv` to prevent accidental overwrites.
- `fase0-audit` creates a structured JSON run log under `<data_root>/logs/` containing parameter metadata, row counts, rejection rates, and SHA-256 artifact hash.
- `fase0-reproducibility` guarantees bit-identical output CSVs and hashes across runs when supplied with the same random `--seed`.

## How to get to it (user POV)

- Run `uv run pred-engine-fase0 <seed_csv> --data-root <path> [--period <int>] [--n-series <int>] [--seed <int>] [--minimo-filas <int>]`.
- Run through harness: `./.cursor/skills/verify-pred-engine/harness.sh drive-fase0 <evidence_dir> [seed_csv] [data_root]`.

## Driving it with harness.sh

Preconditions:

- Runtime environment passes `./.cursor/skills/verify-pred-engine/harness.sh doctor`.
- A valid seed CSV is prepared with required columns `sku_id,timestamp,demand_qty,lead_time_days` and at least `2 * period` observations per SKU (or generated via `harness.sh seed`).
- A clean, disposable data root directory is specified.

- **Execute Phase 0 simulation.** Run `./.cursor/skills/verify-pred-engine/harness.sh drive-fase0 <evidence_dir> <seed_csv> <data_root>` or run directly:
  ```bash
  uv run pred-engine-fase0 /tmp/seed.csv --data-root /tmp/data --period 7 --n-series 2 --minimo-filas 0 --seed 42
  ```
  Exit code is `0`. Stdout prints `Artefacto: <data_root>/raw/panel_sintetico_fase0.csv` and `Bitacora: <data_root>/logs/fase0_<timestamp>_seed42.json`.
- **Confirm WORM storage.** Inspect write permissions on `<data_root>/raw/panel_sintetico_fase0.csv`. The file is write-protected (read-only mode 0444).
- **Confirm audit log integrity.** Read `<data_root>/logs/fase0_<timestamp>_seed42.json`. Confirm `artefacto_sha256` matches the SHA-256 computed from `panel_sintetico_fase0.csv`, `contract_version` equals `"1.0"`, and `row_count` matches the total lines minus header.
- **Confirm bit-identical reproducibility.** Re-run the command with the exact same seed into a second temporary data root. Compute SHA-256 for both output CSVs; both hashes must be strictly identical.
- **Proof capture.** Check `<evidence_dir>/synthetic-data/status.txt`, `<evidence_dir>/synthetic-data/stdout.txt`, `<evidence_dir>/synthetic-data/bitacora.json`, and sample rows in `artifact_head.csv`.

## Gotchas

- Any SKU in the seed file with fewer than `2 * period` rows is skipped; if all SKUs fail this threshold, the command raises a `ValueError` indicating insufficient length.
- The default `--minimo-filas` threshold is 50,000 rows. When running tests or lightweight verification runs on small seed files, always pass `--minimo-filas 0`.
- The CSV output is strictly placed in `<data_root>/raw/`, while run logs are placed in `<data_root>/logs/`. JSON files are forbidden from landing in `raw/`.
- Removing the disposable data root during cleanup requires restoring write permissions (`chmod -R u+w <data_root>`) because of WORM protection.
