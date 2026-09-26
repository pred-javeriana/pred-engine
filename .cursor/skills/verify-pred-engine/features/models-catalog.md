# Provider model catalog

The model catalog lists supported LLM model identifiers for each configured provider (Gemini, OpenAI, Anthropic) within the economic cost tier, displaying the default model and all permitted model choices without requiring an API key, file access, or network connectivity.

## Sub-features

- `models-gemini` lists allowed Google Gemini models and identifies the default lowest-cost model.
- `models-openai` lists allowed OpenAI models and identifies the default lowest-cost model.
- `models-anthropic` lists allowed Anthropic models and identifies the default lowest-cost model.
- `models-invalid` rejects unsupported provider names with an exit code of 1 and an explanatory message.

## How to get to it (user POV)

- Run `uv run pred-engine models` in terminal (defaults to Gemini).
- Run `uv run pred-engine models --provider <provider>` specifying `gemini`, `openai`, or `anthropic`.

## Driving it with harness.sh

Preconditions:

- Python 3.12 environment is initialized and dependencies synced.
- `./.cursor/skills/verify-pred-engine/harness.sh doctor` reports OK.

- **Query Gemini catalog.** Run `./.cursor/skills/verify-pred-engine/harness.sh drive-models <evidence_dir>` or `uv run pred-engine models --provider gemini`. Exit code is `0`, stdout identifies `Proveedor: gemini`, indicates `Default (mas barato): gemini-2.5-flash-lite`, and enumerates permitted models including `gemini-2.5-flash-lite (default)`, `gemini-3.5-flash`, and others.
- **Query OpenAI catalog.** Run `uv run pred-engine models --provider openai`. Exit code is `0`, stdout identifies `Proveedor: openai`, indicates `Default (mas barato): gpt-4.1-nano`, and enumerates permitted models including `gpt-4.1-nano (default)`, `gpt-5-mini`, and `gpt-4o-mini`.
- **Query Anthropic catalog.** Run `uv run pred-engine models --provider anthropic`. Exit code is `0`, stdout identifies `Proveedor: anthropic`, indicates `Default (mas barato): claude-haiku-4-5`, and enumerates permitted models including `claude-haiku-4-5 (default)`.
- **Reject unsupported provider.** Run `uv run pred-engine models --provider invalid-provider`. Exit code is `1` and stderr reports `Proveedor no soportado`.
- **Capture proof.** Run `./.cursor/skills/verify-pred-engine/harness.sh drive-models <evidence_dir>`. Output files in `<evidence_dir>/models/` store stdout for each provider and confirm exit codes in `summary.txt`.

## Gotchas

- Canonical provider names are `gemini`, `openai`, and `anthropic`. Common aliases like `google`, `gpt`, and `claude` are accepted by normalization, but custom names outside this set fail.
- The `models` subcommand is strictly an offline catalog lookup; it does not check whether your API key is valid or has sufficient quota.
- Model IDs must match the catalog when passed to `--model` in subsequent `probe` or `ingest` calls; passing an unlisted model causes validation errors.
