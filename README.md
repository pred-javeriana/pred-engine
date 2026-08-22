# pred-engine

`pred-engine` is the pure-Python analytical library that powers the
**PRED** platform (Plataforma de Evaluación y Recomendación de modelos de
Demanda). It provides the complete forecasting pipeline: L1 ingestion and
series characterisation, L2 model fitting behind a uniform `BaseForecaster`
contract, L3 walk-forward evaluation with statistical comparison tests, and
L4 retrospective validation that produces hold/partial/fail verdicts and
audit-ready export bundles. The library has no UI and no persistence layer;
all inputs and outputs are plain NumPy arrays or pandas DataFrames so that
the `pred-platform` web application can consume them through a clean,
version-stable interface.

PRED is an academic software project developed at Pontificia Universidad
Javeriana under a software-engineering capstone course. It targets a
real-world demand-forecasting use case for a single company, satisfying the
requirements of the *Software Requirements Specification* (SRS v1.0) and the
*Software Project Management Plan* (SPMP). The library is released under the
MIT licence to allow institutional publication.

## Setup

```bash
# Requires uv — https://docs.astral.sh/uv/
uv sync --extra dev
```

## Running tests

```bash
uv run pytest            # runs the suite with an 80% coverage gate
```

## Linting and formatting

```bash
uv run ruff check .      # lint
uv run ruff format .     # format
uv run ruff format --check .  # format check (CI mode)
```

## Pre-commit hooks

```bash
uv run pre-commit install        # install hooks
uv run pre-commit run --all-files  # run manually
```

## L1 ingestion (raw storage)

Passive extraction lives in `pred_engine.ingesta`. Callers inject a
`data_root` (or set `PRED_DATA_ROOT`); the library never writes to a
hard-coded `/data` path.

```
{data_root}/
  raw/         # immutable source CSVs — programmatic writes are blocked
  staging/     # reserved for later cleaning / imputation sessions
  processed/   # Snappy Parquet artefacts for modules 2 and 3
```

```python
from pred_engine.comun.logger import configure_json_logger
from pred_engine.ingesta.data import ensure_data_layout
from pred_engine.ingesta.lector import extract_csv, export_parquet

configure_json_logger("pred_engine")
layout = ensure_data_layout("data")
artefacto = extract_csv(layout.raw / "ventas.csv", data_root=layout.root)
export_parquet(
    artefacto.frame,
    layout.processed / "ventas.parquet",
    data_root=layout.root,
)
```

Structured JSON logs (timestamp, level, module, file hash, row count) go to
stdout. See `docs/features/ingesta-almacenamiento-crudo/` for the API and
session notes.
