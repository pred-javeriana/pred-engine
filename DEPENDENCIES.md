# DEPENDENCIES.md

Dependency manifest for `pred-engine` — updated on every dependency change
(SRS RNF traceability requirement).

## Runtime dependencies

| Package | Version constraint | Purpose |
|---------|-------------------|---------|
| numpy | >=1.26 | N-dimensional arrays; core data structure for all time-series operations |
| pandas | >=2.2 | Tabular I/O for passive CSV extraction and in-memory ingestion artefacts |
| pyarrow | >=16 | Columnar Parquet engine (`engine="pyarrow"`) for processed exports |

> Direct runtime dependencies are added here as each layer is implemented.
> Pydantic schema validation is intentionally deferred to TASK-DATA-1.2-B1.

## Development / test dependencies

| Package | Version constraint | Purpose |
|---------|-------------------|---------|
| pytest | >=8.0 | Test runner for the unit and integration suite |
| pytest-cov | >=5.0 | Coverage measurement; enforces the 80% gate on every CI run |
| ruff | >=0.5 | Linter and formatter (replaces flake8 + isort + black) |
| pre-commit | >=3.7 | Git hook runner; enforces ruff checks before every commit |

## Lock file

Exact pinned versions for all transitive dependencies are recorded in
`uv.lock`, which is committed to the repository (RNF-REP-01: reproducible
execution from a clean clone).
