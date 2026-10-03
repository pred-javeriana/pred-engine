# pred-engine

`pred-engine` is the pure-Python analytical library that powers the
**PRED** platform (Plataforma de Evaluación y Recomendación de modelos de
Demanda). It provides L1 ingestion and series characterisation, L2 model
selection and fitting behind a uniform `BaseForecaster` contract, and causal
L3 walk-forward evaluation. L3 statistical comparison and L4 retrospective
validation (hold/partial/fail verdicts and audit bundles) are not implemented.
The library has no UI or database; ingestion publishes Parquet under a
caller-owned data root. Typed artifacts, NumPy arrays and pandas DataFrames
can be consumed by applications without terminal automation.

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

## Coordinated backend pipeline

`pred_engine.pipeline.Pipeline` owns the four-stage transition path. Existing
stage APIs remain independently callable. The default composition routes the
core classical, ML and DL families using the existing topology matrix;
foundation is opt-in and requires the `foundation` extra.

```python
from pred_engine.pipeline import EvaluationSettings, PipelineInput
from pred_engine.pipeline_setup import build_pipeline

pipeline = build_pipeline(EvaluationSettings(min_train=40), n_trials=4)
result = pipeline.run(PipelineInput(csv_path="sales.csv", data_root="data"))
assert result.blocked_stage == "L4"
assert not result.complete
# Actual L1, L2 and walk-forward L3 artifacts are available on result.
```

Canonical CSV inputs need no LLM. Pass an injectable `provider` for the existing
semantic ingestion path, or `parquet_path` to consume a classified 1.4 artifact.
L1's published contract automatically becomes per-SKU selection requests; L2's
selected factory configurations automatically become causal L3 evaluations.
L3 is diagnostic evidence on the selection history, not an independent
retrospective verdict.

```bash
uv run python -m pred_engine.verify
uv run pred-engine run --csv sales.csv --data-root data --trials 4
```

The offline proof uses real, bounded SARIMA selection and evaluation. Its exit
`0` means the integration checks passed, while its report explicitly says
`complete: false` and `blocked_stage: L4`. The diagnostic `run` command instead
returns `7` for this unfinished-stage blocker (`1` on stage failure). Existing
`pred-engine verify --parquet ...` still verifies only the L1 output contract.

See [backend pipeline API and boundaries](docs/features/backend-pipeline/README.md)
for independent stage execution, errors and the future L4 attachment point.

### Corrida M0 → M1 → M2 desde la semilla

Un solo comando genera el panel sintético de la Fase 0, lo ingiere y clasifica
(L1) y ejecuta la selección y el pronóstico de M2 (L2-L3) en paralelo:

```bash
uv run pred-engine run --seed-csv inventory_data.csv --data-root data \
  --m0-metodo mbb-directo \
  --m0-columna sku_id=Item_ID --m0-columna timestamp=Date \
  --m0-columna demand_qty=Avg_Usage_Per_Day \
  --m0-columna lead_time_days=Restock_Lead_Time
```

M1 clasifica y L2/L3 optimizan solo con la historia anterior a la reserva
cronológica del 20 % (ADR-03-003, ADR-019). Cada SKU × familia es una unidad
aislada y `--workers` (por defecto, núcleos - 1) fija los procesos. La corrida
queda en `data/runs/{run_id}/` con el manifiesto tipado de candidatos para M3
(`candidatos.json`), los pronósticos desde t*, la evidencia walk-forward, la
traza de cada unidad con su modelo y el uso de CPU y memoria de cada proceso;
`pred-engine telemetry data/runs/{run_id}` los grafica. Repetir el comando
retoma los estudios HPO terminados sin reentrenar. El código de salida es `7`
(L4 ausente) u `8` (L4 ausente con unidades fallidas aisladas). Ver
[EJECUCION_M0_M2.md](docs/features/backend-pipeline/EJECUCION_M0_M2.md).

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

## L1.3 SKU topology (Syntetos-Boylan)

After the daily panel exists, `pred_engine.ingesta.categorizacion` computes
ADI and CV² per SKU over its history up to the reserve cut t* (ADR-019) and
injects a single `sku_class` label
(`smooth` | `intermittent` | `erratic` | `lumpy`) on every row of that SKU.
Thresholds are 1.32 and 0.49. Original demand columns are not rewritten.

```bash
uv run pred-engine classify --csv inventory_data.csv --data-root data
```

No LLM key is required. See `docs/features/1.3-motor-enrutador-sku/`.

## L1.4 output contract (Parquet handoff)

After topology, `pred_engine.ingesta.salida` validates the five-column
contract, checks that 1.3 only added `sku_class`, and publishes Snappy
Parquet under `{data_root}/processed/`. Modules 2 and 3 must consume this
artefact, not the raw CSV.

```bash
uv run pred-engine classify --csv local_data/inventory_data.csv --data-root data
uv run pred-engine verify --parquet data/processed/inventory_data.parquet
```

See `docs/features/1.4-artefacto-salida/`.

## Module 2 deep-learning selection

`DLSelectionStrategy` provides a trainable NumPy multilayer predictor and a
bounded architecture/training search through the shared TPE, ASHA and causal
Walk-Forward engine. Register it explicitly as family `dl`; the initial router
policy permits DL only for `smooth` and `erratic` SKUs.

See [DL selection API and validation](docs/features/2.Seleccion-config-modelos/2.6-DLSelectionStrategy/README.md)
for configuration, result evidence and shared recovery behavior.

## Phase 0 pre-ingestion simulation (data augmentation)

`pred_engine.aumentacion` construye el panel sintético con el que se valida
PRED. La semilla se lleva a un calendario diario por SKU y se aumenta con el
método elegido en `--metodo` (ADR-016): `stl-mbb` (STL + MBB de residuales,
ADR-01-007, por defecto) o `mbb-directo` (MBB de la serie diaria, que conserva
la intermitencia de la semilla Kaggle). Cada candidata pasa por las leyes
físicas (demanda entera no negativa, lead time acotado por la semilla) y luego
por la compuerta de paridad del 5 % en media y varianza. El artefacto cumple un
contrato de 4 columnas y se escribe una sola vez en `{data_root}/raw/` (WORM,
modo 0444). El orquestador no importa el framework PRED (módulo 1).

```bash
uv run pred-engine-fase0 inventory_data.csv --data-root data \
  --metodo mbb-directo --columna sku_id=Item_ID --columna timestamp=Date \
  --columna demand_qty=Avg_Usage_Per_Day --columna lead_time_days=Restock_Lead_Time
```

La misma semilla y configuración producen un artefacto idéntico byte a byte;
la bitácora JSON de cada corrida queda en `{data_root}/logs/`. Ver
`docs/features/0.3-0.4-simulacion-fase0/`.
