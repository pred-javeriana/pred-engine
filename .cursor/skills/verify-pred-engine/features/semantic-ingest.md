# Semantic header alignment and ingestion (probe and ingest)

Semantic alignment uses an LLM provider (Gemini, OpenAI, Anthropic) to map chaotic or non-standard CSV headers into the canonical four-column contract (`sku_id`, `timestamp`, `demand_qty`, `lead_time_days`). The `probe` subcommand returns a read-only structured JSON diagnostic, while the `ingest` subcommand executes the entire ingestion lifecycle through schema validation, daily panel resampling, Syntetos-Boylan classification, and Parquet handoff.

## Sub-features

- `probe-diagnostic` inspects arbitrary source CSV headers via an LLM call and outputs a structured diagnostic JSON without creating or modifying local datasets.
- `probe-failclosed` strictly verifies that `probe` refuses to proceed when an API key is absent, exiting cleanly with code 1.
- `ingest-pipeline` executes the full automated pipeline: raw CSV deposition, semantic header alignment, Pydantic row validation, daily grid resampling, SKU classification, and Parquet export.
- `ingest-exit-codes` maps distinct failure modes to explicit exit codes (1: provider/model/arg error, 2: semantic mapping rejection, 3: schema barrier error, 4: LLM timeout, 5: topology error, 6: output handoff error).

## How to get to it (user POV)

- Run `uv run pred-engine probe --csv <path> [--provider <p>] [--model <m>] [--api-key <k>] [--data-root <d>] [--timeout <t>]`.
- Run `uv run pred-engine ingest --csv <path> [--provider <p>] [--model <m>] [--api-key <k>] [--data-root <d>] [--timeout <t>]`.
- Run fail-closed probe check through harness: `./.cursor/skills/verify-pred-engine/harness.sh drive-probe-failclosed <evidence_dir>`.

## Driving it with harness.sh

Preconditions:

- Runtime environment passes `./.cursor/skills/verify-pred-engine/harness.sh doctor`.
- Input CSV file exists.
- For online execution: valid API key supplied via `--api-key` or environment variables (`PRED_LLM_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
- For offline execution: test fail-closed behavior to verify credential enforcement.

- **Drive fail-closed check without API key.** Run `./.cursor/skills/verify-pred-engine/harness.sh drive-probe-failclosed <evidence_dir>` or directly:
  ```bash
  env -u PRED_LLM_API_KEY -u GEMINI_API_KEY uv run pred-engine probe --csv sample.csv --provider gemini
  ```
  Exit code is `1`. Stderr reports `Falta API key. Pase --api-key o defina PRED_LLM_API_KEY / ...`. No network calls are made.
- **Probe with LLM credentials.** When an API key is available:
  ```bash
  uv run pred-engine probe --csv messy_inventory.csv --provider gemini --api-key "$GEMINI_API_KEY"
  ```
  Exit code is `0`. Stdout prints `estado: accepted` or `estado: rejected`, followed by provider, model, and diagnostic JSON details.
- **Execute complete ingestion with LLM credentials.** When an API key is available:
  ```bash
  uv run pred-engine ingest --csv messy_inventory.csv --provider gemini --api-key "$GEMINI_API_KEY" --data-root /tmp/data
  ```
  Exit code is `0`. Stdout prints `filas_crudas`, `filas_validadas`, `filas_panel_diario`, topology metrics table, `contrato_1_4: accepted`, and destination Parquet path under `<data-root>/processed/`.
- **Proof capture.** Check `<evidence_dir>/probe_failclosed/status.txt`, `<evidence_dir>/probe_failclosed/stderr.txt` confirming exit code `1` and exact error message.

## Gotchas

- In `probe`, an unmappable schema produces `estado: rejected` with diagnostic JSON on stdout, but the process exit code is `0` because diagnosing invalid input is a successful probe run.
- In `ingest`, an unmappable schema produces an error and exits with code `2` (`SemanticAlignmentError`), preventing partial or corrupt ingestions.
- When passing `--model`, ensure the model name exists in the allowed economic tier catalog for that provider (see `pred-engine models --provider <name>`).
- The API key is never printed or logged in JSON loggers or stdout.
