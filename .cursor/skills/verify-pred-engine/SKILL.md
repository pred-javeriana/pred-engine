---
name: verify-pred-engine
description: Verify the pred-engine analytical command-line interface (models, probe, ingest, classify, verify) and the pred-engine-fase0 synthetic data simulation tool through isolated execution, verifiable file side effects, and structured log audits.
---

# Verify PRED Engine

Read [features/README.md](features/README.md) and the relevant feature map file before driving. PRED Engine is a Python command-line analytical engine and pre-ingestion simulation tool, not an HTTP daemon or browser UI. Its user surface consists of two CLI entry points:
1. `pred-engine` with subcommands `models`, `probe`, `ingest`, `classify`, and `verify`.
2. `pred-engine-fase0` for Phase 0 synthetic panel generation (Moving Block Bootstrap + rejection sampling).

Never drive with shared or production data paths (`/data` or `data/` in repo root). Always execute with an isolated, disposable data directory (`--data-root /tmp/pred-verify-$RUN_ID/data`).

## Launch

For this short-lived CLI surface, there is no long-running server daemon to keep alive. Launch consists of syncing the Python 3.12 virtual environment and ensuring the CLI entry points are compiled and ready on `PATH`:

```bash
ROOT="$(pwd -P)"
RUN_ID="$(date +%s)-$RANDOM"
EVIDENCE_DIR="${TMPDIR:-/tmp}/pred-verify-$RUN_ID/evidence"
DISPOSABLE_DATA_ROOT="${TMPDIR:-/tmp}/pred-verify-$RUN_ID/data"
mkdir -p "$EVIDENCE_DIR" "$DISPOSABLE_DATA_ROOT"

# Ensure dependencies are present
uv sync
```

Readiness is proven when `uv run pred-engine --help` and `uv run pred-engine-fase0 --help` exit with status 0. Teardown consists of removing `$DISPOSABLE_DATA_ROOT` while retaining `$EVIDENCE_DIR`.

## Doctor

Run this read-only diagnostic check before driving any feature or whenever behavior appears anomalous:

```bash
./.cursor/skills/verify-pred-engine/harness.sh doctor
```

The Doctor verifies:
1. `uv` executable is on `PATH`.
2. Python runtime is `>= 3.12`.
3. Core analytics libraries (`numpy`, `pandas`, `pyarrow`, `pydantic`, `statsmodels`) import without error.
4. `pred-engine` CLI is functional and exposes subcommands: `models`, `probe`, `ingest`, `classify`, `verify`.
5. `pred-engine-fase0` CLI is functional and exposes expected simulation parameters.

If any check fails, do not proceed with driving until the environment is restored.

## Drive

Drive features through the project CLI entry points or via the `./.cursor/skills/verify-pred-engine/harness.sh` helper.

### 1. Model catalog (`models`)

Query allowed LLM models for a provider without requiring API keys or network calls:

```bash
./.cursor/skills/verify-pred-engine/harness.sh drive-models "$EVIDENCE_DIR"
# Or directly:
uv run pred-engine models --provider gemini
```

### 2. Phase 0 synthetic simulation (`pred-engine-fase0`)

Run the pre-ingestion simulation to produce a synthetic panel under WORM protection:

```bash
# Generate seed if not present
SEED_CSV="$EVIDENCE_DIR/seed.csv"
./.cursor/skills/verify-pred-engine/harness.sh seed "$SEED_CSV" 2 30

# Execute Phase 0 simulation
./.cursor/skills/verify-pred-engine/harness.sh drive-fase0 "$EVIDENCE_DIR" "$SEED_CSV" "$DISPOSABLE_DATA_ROOT"
# Or directly:
uv run pred-engine-fase0 "$SEED_CSV" \
  --data-root "$DISPOSABLE_DATA_ROOT" \
  --period 7 --n-series 2 --minimo-filas 0 --seed 42
```

### 3. SKU topology classification (`classify`)

Classify demand patterns into Syntetos-Boylan quadrants and export Snappy Parquet:

```bash
INPUT_CSV="$DISPOSABLE_DATA_ROOT/raw/panel_sintetico_fase0.csv"
./.cursor/skills/verify-pred-engine/harness.sh drive-classify "$EVIDENCE_DIR" "$INPUT_CSV" "$DISPOSABLE_DATA_ROOT"
# Or directly:
uv run pred-engine classify --csv "$INPUT_CSV" --data-root "$DISPOSABLE_DATA_ROOT"
```

### 4. Output Parquet contract verification (`verify`)

Audit the 1.4 output contract on a published Parquet file:

```bash
PARQUET_FILE="$DISPOSABLE_DATA_ROOT/processed/panel_sintetico_fase0.parquet"
./.cursor/skills/verify-pred-engine/harness.sh drive-verify "$EVIDENCE_DIR" "$PARQUET_FILE"
# Or directly:
uv run pred-engine verify --parquet "$PARQUET_FILE"
```

### 5. Semantic alignment and ingestion (`probe` / `ingest`)

Validate fail-closed security when credentials are absent:

```bash
./.cursor/skills/verify-pred-engine/harness.sh drive-probe-failclosed "$EVIDENCE_DIR"
```

When valid LLM credentials are provided via `--api-key` or `PRED_LLM_API_KEY`:

```bash
uv run pred-engine probe --csv "$SEED_CSV" --provider gemini --api-key "$GEMINI_API_KEY"
uv run pred-engine ingest --csv "$SEED_CSV" --provider gemini --api-key "$GEMINI_API_KEY" --data-root "$DISPOSABLE_DATA_ROOT"
```

## Evidence

Store all execution transcripts, status files, and side-effect artifacts in `$EVIDENCE_DIR`. Proof standards require:
1. **Action and resulting state:** Capture the exact invocation, exit code, stdout, and stderr.
2. **Side-effect assertions:**
   - For `pred-engine-fase0`: Verify the existence of `<data_root>/raw/panel_sintetico_fase0.csv`, confirm WORM read-only permissions (mode 0444), and verify the SHA-256 match in `<data_root>/logs/fase0_*.json`.
   - For `pred-engine classify`: Verify the generated Snappy Parquet file at `<data_root>/processed/<stem>.parquet`, confirm `contrato_1_4: accepted` on stdout, and check row count matching panel size.
   - For `pred-engine verify`: Verify `contrato_1_4: accepted` and that all five contract columns (`sku_id`, `timestamp`, `demand_qty`, `lead_time_days`, `sku_class`) are present without nulls.
   - For fail-closed checks: Verify exit code 1 and stderr message `Falta API key`.

Evidence files must survive cleanup and remain accessible at `$EVIDENCE_DIR` after teardown.

## Cleanup

Tear down only the disposable data roots and scratch workspaces created during the verification run. Never delete the evidence directory:

```bash
# Unlock WORM permissions before deletion
./.cursor/skills/verify-pred-engine/harness.sh cleanup "$DISPOSABLE_DATA_ROOT"

# Confirm evidence survival
test -d "$EVIDENCE_DIR" && ls -la "$EVIDENCE_DIR"
```

## Helpers

The skill includes the harness script `./.cursor/skills/verify-pred-engine/harness.sh`. It is fully executable and supports:
- `harness.sh doctor`: Verifies the runtime environment and CLI subcommands.
- `harness.sh seed <path> [skus] [days]`: Creates a canonical 4-column CSV dataset for testing.
- `harness.sh drive-models <evidence_dir>`: Runs catalog queries across all supported providers.
- `harness.sh drive-fase0 <evidence_dir> [seed] [data_root]`: Runs Phase 0 simulation and captures WORM/audit evidence.
- `harness.sh drive-classify <evidence_dir> <csv> [data_root]`: Runs topology classification and verifies Parquet output.
- `harness.sh drive-verify <evidence_dir> <parquet>`: Audits Parquet contract compliance.
- `harness.sh drive-probe-failclosed <evidence_dir>`: Proves fail-closed security for unauthenticated probe calls.
- `harness.sh cleanup <disposable_path>`: Unlocks and removes disposable data root directories.
