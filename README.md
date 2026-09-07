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

## L1.2 semantic alignment

After passive extraction, `pred_engine.ingesta.pipeline.run_ingest` maps chaotic
headers via an injectable LLM provider (Gemini, OpenAI, or Anthropic), validates
rows with Pydantic, and resamples each SKU onto a daily grid (demand gaps → 0).

Each provider exposes a curated list of cost-tier models (`AVAILABLE_MODELS` in
`pred_engine.comun.llm.catalogo`). If `--model` is omitted, the cheapest default
is used; use `pred-engine models --provider <name>` to see allowed IDs.

```bash
uv run pred-engine models --provider gemini
uv run pred-engine ingest \
  --csv inventory_data.csv \
  --provider gemini \
  --model gemini-3.5-flash \
  --data-root data
```

The API key is read from `--api-key` or `PRED_LLM_API_KEY` / `GEMINI_API_KEY`
(and equivalents for OpenAI and Anthropic). If the probe cannot map `sku_id`,
`timestamp`, `demand_qty` and `lead_time_days` with confidence, ingestion stops.
See `docs/features/1.2-alineacion-semantica-validacion/`.

## Phase 0 pre-ingestion simulation (data augmentation)

`pred_engine.aumentacion` builds the synthetic stress panel that PRED is
validated against. Starting from an STL + Moving Block Bootstrap of a Kaggle
seed (`aumentacion.mbb`), it enforces logistics conservation laws (non-negative
integer demand, seed-derived lead-time bounds), runs basic rejection sampling so
each synthetic series stays within 5% of the seed's mean and variance, validates
a strict 4-column data contract, and writes a single immutable CSV to
`{data_root}/raw/` under a Write-Once-Read-Many guard. The orchestrator is
One-Shot and does not import the PRED framework (module 1).

```bash
uv run python -m pred_engine.aumentacion.fase0 seed_kaggle.csv \
  --data-root data --period 7 --n-series 40 --seed 42
```

Runs with the same `--seed` produce a byte-identical artefact; a structured JSON
run log lands in `{data_root}/logs/`. See
`docs/features/0.3-0.4-simulacion-fase0/`.
