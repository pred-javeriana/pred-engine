# Coordinated backend pipeline

`pred_engine.pipeline.Pipeline` owns the fixed L1 -> L2 -> L3 -> L4 transition
path. Library callers, the CLI and the bounded offline diagnostic use that
same application service. Phase 0 is an optional input producer, not a fifth
principal stage.

## Stages

- **L1:** canonical or semantic CSV ingestion, daily resampling, topology
  classification, cutoff t* computed once from the full CSV calendar, per-product
  late-launch exclusion and the five-column immutable Parquet contract. The cutoff
  t* and exclusions are stored in schema key-value metadata (`pred_engine`). A
  classified Parquet without this metadata is refused with an error to re-import
  from the CSV. In `Pipeline.ingest`, `classify --parquet` and
  `ingestion_artifact_from_publication`, one read of the published file supplies
  its rows, SHA-256, t*, exclusions and topology, so a file replaced at the same
  path does not mix two publications. `Pipeline.fit` refuses an ingestion
  artifact without its cut instead of reading t* again from the path.
- **L2:** topology routing, per-family HPO, fitting on history <= t* (read directly
  from file metadata) and the forecast from t*. Excluded products from L1 are
  recorded as 'sin historia admisible: lanzado despues del corte'. Each SKU/family
  is an isolated unit. The families are SARIMA, LightGBM, NumPy MLP and Chronos-2.
  Chronos-2 runs by default (`--no-foundation` drops it), is CPU-only and requires
  the `foundation` extra and fixed local weights. When L2 closes, a SKU with two
  or more routed families and exactly one evaluable family leaves the
  competition with `eligibility.fewer_than_two_families` (ADR-023); it reaches
  neither L3 nor M3. A SKU with no evaluable family goes on to M3 as
  FALLO_TECNICO.
- **L3:** causal walk-forward evaluation on the same admissible history used
  by HPO. Each window has a new fit. `selection_scope="same_history"` makes
  the limitation explicit. These diagnostic metrics are not a holdout verdict.
- **L4:** native M3 through `ControladorEvaluacion`: strict M2 handoff,
  complete daily reserved windows split into a selection block and a test
  block, category selection frozen on the selection block, five descriptive
  verdicts, complementary HLN-DM with BH within each class and the champion
  confirmation on the test block, and durable publication to the central
  platform database.

L2/L3 retain independent process units with `workers > 1`. A unit failure
preserves the rest of its stage; no successful unit means a stage error.
L4 forecasts each SKU in a pool of `procesos` processes; the main process
stays the only SQLite writer and confirms the units in SKU order.
M3 records forecast failures per origin without substituting zeros.

## Composition

```python
from pred_engine.forecasting.control_reanudacion import ControladorEvaluacion
from pred_engine.forecasting.persistencia_evidencia import abrir_base_plataforma
from pred_engine.pipeline import EvaluationSettings, PipelineInput
from pred_engine.pipeline_setup import build_pipeline
from pred_engine.run_artifacts import code_fingerprint

# workers > 1 starts spawned processes, so a script needs this guard.
if __name__ == "__main__":
    with abrir_base_plataforma("data/pred.db") as conn:
        validator = ControladorEvaluacion(
            conn, "data/runs/sales", run_id_m2="sales",
            codigo_version=code_fingerprint(), datos_sinteticos=False,
            data_root="data", procesos=4,
        )
        pipeline = build_pipeline(
            EvaluationSettings(min_train=60, horizon=7, step=28),
            families=("classical", "ml"), n_trials=10, workers=4,
            hpo_root="data/runs/sales/hpo", session="sales", validator=validator,
        )
        result = pipeline.run(
            PipelineInput(csv_path="sales.csv", data_root="data", datos_sinteticos=False)
        )
```

The platform migrator creates the database first. The engine opens it in
read/write-existing mode and applies pending M3 migrations, never another
standalone database. The same database and schema version serve platform and
CLI runs. Inputs select exactly one CSV or classified Parquet. Semantic
probing remains an injected provider for CSV inputs only.

`ConfiguredTopologyPolicy` filters an explicit subset of the existing matrix.
It never routes ML/DL for lumpy SKUs or DL for intermittent SKUs. A configured
subset with no eligible family fails explicitly. `n_trials=None` preserves
the original strategy budgets and never silently reduces model capacity.

## Independent stages

```python
ingestion = pipeline.ingest(request)
fitting = pipeline.fit(ingestion)
evaluation = pipeline.evaluate(fitting)
validation = pipeline.validate(evaluation)
```

`FittingArtifact` contains the reserve cut, selected recipes, admissible
history, forecasts from t*, exclusions and unit records. `EvaluationArtifact`
contains the causal L3 windows and settings. Native L4 consumes all M2 fitted
candidates, not just those whose L3 reevaluation completed.

Cada exclusión y cada falla de unidad lleva un código estable (ADR-022):
`eligibility.*` en `exclusions[].code`, y el código de la excepción en
`failures[].code`, `unidades.jsonl`, el evento `unit_finished` y
`seleccion.jsonl`. El texto (`cause`, `error`) sigue siendo el detalle técnico.

`ValidationArtifact` exposes one `SkuVerdict` per handed-off SKU and an
`audit_bundle` with M3 identity, publication, coverage and export paths.
Native verdicts are FALLO_TECNICO, NO_EVALUABLE, EVIDENCIA_INSUFICIENTE,
VALIDADO and EXPLORATORIO. The legacy hold/partial/fail vocabulary remains
accepted only for existing caller-provided validators; native M3 never uses it.

Applications can still inject their own router, factories and
`RetrospectiveValidator`. A caller-provided validator is marked as such.
A library pipeline without a validator leaves L4 blocked by missing database
configuration; it does not claim evaluation occurred. The CLI always composes
native M3 when its database prerequisite is satisfied.

## Results and failures

`PipelineResult` contains exactly four stage records with pending, completed,
blocked or failed states. `complete` means all configured operations completed,
not that every model won or every SKU had sufficient evidence.
`PipelineExecutionError` retains the failed stage, completed outputs and
original cause. `PipelineResult.failures` retains isolated L2/L3 failures;
native M3's audit bundle also reports technical forecast failures.

The CLI returns 0 for complete publication, 8 for publication with isolated
technical failures and 1 for stage, identity or database failure. A rejected
handoff is stored as RECHAZADA before reserved windows run. Statistical losses
against the baseline are empirical results, not technical process failures.

## Recovery and publication

M2 reuses its HPO manifests and completed trials. M3 freezes its protocol and
handoff before reserved observations are used. It confirms result, marker and
failure cause together for each origin. A restart loads confirmed forecasts
and executes only pending origins, then resumes evaluation or publication from
the durable run phase. The M2 handoff emission does not change on resume. A
failed L2 unit runs again on resume, so the retry can change which SKUs the
two-family rule excludes; M3 keeps the exclusions of its durable provenance.

The central database is authoritative. CSV and Parquet are complete derived
exports prepared before the final directory rename. Platform readers expose
only PUBLICADA evidence, so a stopped run never appears as a completed result.
See [M3](../3.Evaluacion/README.md), its API pages, ADR-021 and ADR-023 for exact contracts.

## Diagnostics and verification

`python -m pred_engine.verify` remains a bounded offline L1-L3 diagnostic with
real SARIMA fitting and no platform database. It explicitly leaves L4
unconfigured. Its zero exit means its diagnostic checks passed, not that M3
was executed. Full M3 verification requires a platform database and the real
`pred-engine run` command, followed by independent SQLite/platform readback.

The tests cover input contracts, native model parity, reserve isolation,
parallel units, checkpoints, native M3, interrupted commits, publication and
caller-provided validators. Tests do not replace the coordinated live run of
both installed repositories against a run-scoped platform database.
