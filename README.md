# pred-engine

`pred-engine` is the pure-Python analytical library that powers the
**PRED** platform (Plataforma de Evaluación y Recomendación de modelos de
Demanda). It provides L1 ingestion and series characterisation, L2 model
selection and fitting behind a uniform `BaseForecaster` contract, and causal
L3 walk-forward evaluation. Native L4 runs M3's reserved-window forecasts,
category selection, five descriptive verdicts and complementary HLN-DM/BH.
The library has no UI. M3 writes through an injected connection to the single
platform SQLite database and publishes derived Parquet/CSV. Ingestion retains
a caller-owned data root. Typed artifacts, NumPy arrays and pandas DataFrames
remain usable without terminal automation.

PRED is an academic software project developed at Pontificia Universidad
Javeriana under a software-engineering capstone course. It targets a
real-world demand-forecasting use case for a single company, satisfying the
requirements of the *Software Requirements Specification* (SRS v1.0) and the
*Software Project Management Plan* (SPMP). The library is released under the
MIT licence to allow institutional publication.

## Setup

```bash
# Requires uv — https://docs.astral.sh/uv/
uv sync --frozen --extra dev
```

## Running tests

```bash
uv run pytest                  # runs the suite with an 80% coverage gate
uv run pytest -m "not slow"    # fast suite with coverage gate (CI mode)
uv run pytest -m slow --no-cov # slow tests (benchmarks / model fitting)
```

### Equipos con pocos núcleos (4 núcleos, 8 GB)

La sesión de pruebas ya limita la CPU: `tests/conftest.py` fija una hebra de
BLAS/OpenMP por proceso, y las pruebas que corren la CLI pasan `--workers 1` o
`2`. Medido con 4 núcleos visibles, `pytest -m "not slow"` usa como máximo
unos 2 núcleos y `pytest -m slow` unos 2. Para no saturar el equipo:

1. Corra un solo `pytest` a la vez.
2. Mientras desarrolla, corra solo las pruebas del área que cambió, sin
   cobertura. `tests/` replica la estructura de `src/pred_engine/`:

   ```bash
   uv run pytest -m "not slow" --no-cov tests/optimizacion/HPO
   ```

3. Antes de abrir el PR, corra la corrida de humo de CI (8 SKU, unos 35 s y
   menos de 3 núcleos). Necesita una copia de `pred-platform` para crear la
   base central:

   ```bash
   PYTHONPATH=../pred-platform/src uv run --no-sync python -m pred_platform.dal.migrate \
     --db local_data/humo/pred.db
   uv run pred-engine run --seed-csv tests/fixtures/semilla_inventario.csv \
     --data-root local_data/humo --db-path local_data/humo/pred.db --run-id humo --workers 2 \
     --m0-metodo mbb-directo --m0-n-series 1 --m0-minimo-filas 0 \
     --m0-columna sku_id=Item_ID --m0-columna timestamp=Date \
     --m0-columna demand_qty=Avg_Usage_Per_Day --m0-columna lead_time_days=Restock_Lead_Time
   ```

   El código de salida esperado es `0` u `8`. Pase siempre `--workers 2` en un
   equipo pequeño: el valor por defecto (núcleos - 1) ocupa todos los núcleos.

CI corre las suites completas en cada PR, así que no hace falta repetirlas
localmente.

## Linting, formatting and type checking

```bash
uv run ruff check .           # lint
uv run ruff format .          # format
uv run ruff format --diff .   # format check (CI mode)
uv run pyright                # type check
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
from pred_engine.forecasting.control_reanudacion import ControladorEvaluacion
from pred_engine.forecasting.persistencia_evidencia import abrir_base_plataforma
from pred_engine.pipeline import EvaluationSettings, PipelineInput
from pred_engine.pipeline_setup import build_pipeline
from pred_engine.run_artifacts import code_fingerprint

# The platform creates this database first with its migration command.
with abrir_base_plataforma("data/pred.db") as conn:
    validator = ControladorEvaluacion(
        conn, "data/runs/sales", run_id_m2="sales",
        codigo_version=code_fingerprint(), datos_sinteticos=False, data_root="data",
    )
    pipeline = build_pipeline(
        EvaluationSettings(min_train=40), n_trials=4, validator=validator,
    )
    result = pipeline.run(PipelineInput(csv_path="sales.csv", data_root="data", datos_sinteticos=False))
    assert result.complete
    assert result.validation.audit_bundle["estado"] == "PUBLICADA"
```

Canonical CSV inputs need no LLM. Pass an injectable `provider` for the existing
semantic ingestion path, or `parquet_path` to consume a classified 1.4 artifact.
L1's published contract automatically becomes per-SKU selection requests; L2's
selected factory configurations automatically become causal L3 evaluations.
L3 is diagnostic evidence on the selection history, not an independent
retrospective verdict.

```bash
uv run python -m pred_engine.verify
uv run pred-engine run --csv sales.csv --data-root data --db-path data/pred.db --trials 4
```

The bounded offline proof exercises L1-L3 without requiring a platform database.
Its report keeps L4 unconfigured rather than inventing a validation result.
The real `run` command requires the existing platform database and configures
native M3. It returns `0` after publication, `8` with isolated technical failures,
or `1` on a stage or database error. It applies pending M3 migrations itself,
but never creates a separate database. `PRED_DB_PATH` selects the platform path
when `--db-path` is absent. Mark augmented inputs with `--datos-sinteticos`;
`--seed-csv` does so automatically. Existing `verify --parquet` remains an L1
contract check.

See [backend pipeline API and boundaries](docs/features/backend-pipeline/README.md)
for independent stages and errors. See [M3](docs/features/3.Evaluacion/README.md)
for the reserve protocol, migrations, recovery, statistical limits and delivery to M4.

### Corrida M0 -> M1 -> M2 -> M3 desde la semilla

Un solo comando genera el panel sintético de la Fase 0, lo ingiere y clasifica
(L1), ejecuta M2 en paralelo y publica la evaluación reservada M3 en la base
central existente de plataforma:

```bash
uv run pred-engine run --seed-csv inventory_data.csv --data-root data --db-path data/pred.db \
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
retoma los estudios HPO y los orígenes M3 ya confirmados. Los resultados M3
quedan en SQLite y en exportaciones derivadas. El código de salida es `0` o
`8` cuando la publicación tiene fallos aislados. Ver
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
  ingestas/    # one <sha256>.json report per ingestion (pred-engine.ingesta/1)
```

### Deposit into `raw/` and the ingestion report

`pred_engine.ingesta.data.deposit_raw` is the only writer of `raw/` and it is
add-only: a file already there is never modified or deleted. The same name with
the same content is reused (`already_present_identical`). The same name with
**different** content raises `RawConflictError` and writes nothing; with
`on_conflict="keep_both"` (CLI: `--confirm-overwrite`) the new file is stored as
`<name>-<first 8 chars of its SHA-256><ext>` and both are kept. `pred-engine
classify`/`ingest` exit with code 7 on an unconfirmed conflict.

`ingest_csv_with_report` (used by `classify --csv`, `ingest` and `run`) always
leaves `ingestas/<sha256>.json`, also when the ingestion is `rejected`, `failed`
or `needs_confirmation`: source hash, deposit, header diagnostic, quality log,
published Parquet, reserve cut t\* and the per-SKU topology metrics (ADI, CV²,
class). `pred-engine run` also saves the metrics as `topologia.json` in the run
folder. The platform imports these files; see section 17 of the frontend-engine
contract in `pred-docs`.

### Progress stream of a run

While a run executes, `pred-engine run` appends one JSON line per event to
`runs/{run_id}/progreso.jsonl` (schema `pred-engine.progreso/1`, section 18 of the contract):
`run_started` (with the attempt number), `stage_started`/`stage_finished`, `planned` (the
SKU × family units of a stage, still pending), `excluded`, `unit_started`/`unit_finished` (written
by the worker process at the instant they happen, with state, seconds and error) and
`run_finished` (written after `corrida.json`). Repeating a run with the same `run_id` resumes it:
the stream keeps earlier attempts and the stored HPO studies are reused. The platform follows this
file to show tasks moving from pending to running to done.

### Selection per SKU of a run

`runs/{run_id}/seleccion.jsonl` (schema `pred-engine.seleccion/1`, section 7.5 of the contract)
holds one line per SKU with the four families, whether each was selected, failed, could not run
or was left out (by the policy matrix or by the families the run was configured with), the
evidence of the chosen configuration, its HPO trials (completed, pruned and failed, each with its
reason) and the Walk-Forward result per window. It is written as soon as L2 finishes, even if L3
fails. The trials are read from `hpo/` without Optuna:

```python
from pred_engine.comun.ensayos_hpo import leer_ensayos

for ensayo in leer_ensayos("data/runs/r-.../hpo/classical-FIL-0001-r-..."):
    print(ensayo.estado, ensayo.valor, ensayo.motivo)
```

The foundation family has no study: it is not optimised, so it has neither trials nor counters.

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
stdout. See `docs/features/1.Ingesta/1.1-ingesta-almacenamiento-crudo/` for the API and
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
See `docs/features/1.Ingesta/1.2-alineacion-semantica-validacion/`.

## L1.3 SKU topology (Syntetos-Boylan)

After the daily panel exists, `pred_engine.ingesta.categorizacion` computes
ADI and CV² per SKU over its history up to the reserve cut t* (ADR-019) and
injects a single `sku_class` label
(`smooth` | `intermittent` | `erratic` | `lumpy`) on every row of that SKU.
Thresholds are 1.32 and 0.49. Original demand columns are not rewritten.

```bash
uv run pred-engine classify --csv inventory_data.csv --data-root data
```

No LLM key is required. See `docs/features/1.Ingesta/1.3-motor-enrutador-sku/`.

## L1.4 output contract (Parquet handoff)

After topology, `pred_engine.ingesta.salida` validates the five-column
contract, checks that 1.3 only added `sku_class`, and publishes Snappy
Parquet under `{data_root}/processed/`. Modules 2 and 3 must consume this
artefact, not the raw CSV.

```bash
uv run pred-engine classify --csv local_data/inventory_data.csv --data-root data
uv run pred-engine verify --parquet data/processed/inventory_data.parquet
```

See `docs/features/1.Ingesta/1.4-artefacto-salida/`.

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
