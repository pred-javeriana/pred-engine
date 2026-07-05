# DEPENDENCIES.md

Dependency manifest for `pred-engine` — updated on every dependency change
(SRS RNF traceability requirement).

## Runtime dependencies

| Package | Version constraint | Purpose |
|---------|-------------------|---------|
| numpy | (transitive via dev/test stack) | N-dimensional arrays; core data structure for all time-series operations |

> The library currently has no direct runtime dependencies beyond the Python
> standard library and NumPy (pulled in transitively).  Direct runtime
> dependencies will be added here as each forecasting model integration is
> implemented.

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
