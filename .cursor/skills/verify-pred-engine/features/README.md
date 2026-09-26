# PRED Engine verification map

This directory is the maintained source for verifying the user-facing behavior of `pred-engine` and its companion simulation tool `pred-engine-fase0`. Read this index before driving the application, then use the matching feature file as the execution recipe.

## Baseline preconditions

- Ensure the local Python 3.12 environment is synchronized with `uv sync`.
- Use a dedicated, disposable data root directory (e.g. `/tmp/pred-verify-$RUN_ID/data`) for all data persistence (`--data-root`).
- Never write to or mutate the repository's root `data/` directory during verification runs.
- Run `./.cursor/skills/verify-pred-engine/harness.sh doctor` and require all runtime and CLI entry points to report OK before driving any feature.
- When driving offline commands (`models`, `classify`, `verify`, and `pred-engine-fase0`), no LLM credentials are required.
- When driving online LLM commands (`probe` and `ingest`), ensure an API key is available via `--api-key` or standard environment variables (`PRED_LLM_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`), or test against their documented fail-closed behavior.

## Driving conventions

- Treat every command and option as literal.
- Execute CLI commands via `uv run pred-engine <subcommand>` or `uv run pred-engine-fase0`, or use the `./.cursor/skills/verify-pred-engine/harness.sh` helper.
- Verify side effects directly: inspect generated CSVs under `<data_root>/raw/`, Parquet tables under `<data_root>/processed/`, and audit JSON logs under `<data_root>/logs/`.
- Ensure proof artifacts survive post-test cleanup by writing evidence to a designated evidence directory outside the disposable data root.

## Proof and skip reporting

- Capture the exact command invoked, stdout, stderr, and the process exit code.
- For data transformations, capture before-and-after schema descriptions, row counts, and output artifact paths.
- For `fase0`, verify the WORM (Write-Once-Read-Many) filesystem permissions on the generated raw CSV and check the matching SHA-256 hash in the JSON run log.
- For `classify` and `verify`, confirm that stdout reports `contrato_1_4: accepted` alongside the computed Syntetos-Boylan class distribution (`smooth`, `intermittent`, `erratic`, `lumpy`).
- When a path requires external LLM credentials that are absent, verify the fail-closed rejection and report the credential precondition honestly without claiming mocked external services as real LLM executions.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the user-visible behavior. It then uses exactly four H2 sections in this order:

1. `Sub-features` lists short IDs with one line for each behavior.
2. `How to get to it (user POV)` lists every user entry point.
3. `Driving it with harness.sh` starts with `Preconditions:` and uses labeled bullets that pair each user action with an exact command and observable result.
4. `Gotchas` lists traps that can waste or invalidate a verification run.

Keep implementation details out of the map. Name only user paths, stable handles, required state, commands, and observable proof.

## Features

- [Provider model catalog](models-catalog.md) covers query of economic-tier models across Gemini, OpenAI, and Anthropic providers.
- [Pre-ingestion synthetic data simulation](synthetic-data.md) covers Phase 0 STL + Moving Block Bootstrap simulation, logistics conservation, WORM storage, and audit logging.
- [SKU topology classification](classify.md) covers Syntetos-Boylan quadrant classification (ADI and CV²) on canonical CSV or Parquet and Snappy Parquet export.
- [Output Parquet contract verification](verify.md) covers independent audit and integrity checking of published 1.4 Parquet datasets.
- [Semantic header alignment and ingestion](semantic-ingest.md) covers LLM-assisted header alignment diagnostics (`probe`) and full automated pipeline ingestion (`ingest`).
