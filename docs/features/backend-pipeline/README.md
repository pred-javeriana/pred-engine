# Coordinated backend pipeline

## Repository implementation state

- **L1, implemented:** `ingesta.pipeline` extracts/validates canonical CSVs,
  optionally probes semantics through an injected LLM, resamples daily,
  characterises SKUs and publishes the validated five-column Parquet contract.
- **L2, implemented:** topology routing delegates to classical SARIMA, ML
  LightGBM, DL NumPy MLP or frozen foundation selection. The first three use
  shared HPO; foundation is zero-shot and requires optional dependencies and
  pretrained weights for inference. All four have real forecasters.
- **L3, partial:** `comun.walkforward.evaluar_walk_forward` performs causal
  window refits and computes MAE/RMSE/sMAPE/MASE. The forecasting generators,
  comparison and Diebold-Mariano namespaces are currently empty. The pipeline
  runs the working walk-forward capability and explicitly records L3's partial
  maturity; it does not invent statistical comparison evidence.
- **L4, absent:** the retrospective analysis, testing and report-generation
  namespaces are empty. No built-in verdict or audit-bundle implementation
  exists. The pipeline stops at L4 with no L4 output.

Phase 0 augmentation is an input producer, not a fifth principal stage.

> **Ejecución M0 → M1 → M2 (ADR-017).** L2 y L3 trabajan solo con la historia
> anterior al corte t* de la reserva del 20 % (ADR-03-003) y corren cada SKU ×
> familia como unidad aislada, en paralelo con `workers > 1`. `pred-engine run`
> persiste cada corrida en `{data_root}/runs/{run_id}/` y reutiliza el control
> de reanudación de 2.9 para los estudios HPO. Ver
> [EJECUCION_M0_M2.md](EJECUCION_M0_M2.md) para comandos, artefactos y
> reanudación.

## Application boundary

`pred_engine.pipeline` contains the execution semantics. `Pipeline.run` is the
only transition loop. No CLI, HTTP server, queue or workflow platform is needed.
The existing CLI and `pred_engine.verify` are two callers of this service, not
implementations of stage transitions.

```python
from pred_engine.pipeline import EvaluationSettings, PipelineInput
from pred_engine.pipeline_setup import build_pipeline

pipeline = build_pipeline(
    EvaluationSettings(min_train=40, horizon=7, step=7, seasonality=7),
    families=("classical", "ml", "dl"),
    n_trials=4,
    seed=42,
)
run = pipeline.run(PipelineInput(csv_path="sales.csv", data_root="data"))
```

`n_trials=None` preserves the original per-profile strategy budgets. A run
requires sufficient history for the configured search spaces and at least four
HPO windows; it never silently reduces model capacity to accommodate short data.
`ConfiguredTopologyPolicy` filters an explicitly configured subset of the
existing initial matrix without changing that matrix. For example, it does not
route DL for lumpy or intermittent SKUs. A subset with no permitted family for
an encountered SKU fails at L2 rather than skipping that SKU. Foundation can be
included explicitly. Si su dependencia opcional o sus pesos no están
disponibles, cada unidad fundacional falla en L2 (al pronosticar desde t*) y
queda registrada como falla aislada.

Inputs select exactly one source:

- `PipelineInput(csv_path=..., data_root=...)`: existing canonical CSV path.
- Add `provider=...` and optionally `timeout=...`: existing semantic ingestion.
- `PipelineInput(parquet_path=...)`: read and validate an already classified
  artifact, without repeating raw extraction or characterisation.

L1 always reads the published 1.4 Parquet contract before downstream use. L2
calcula el corte t* (`ReserveCut`), recorta cada SKU a su historia admisible,
excluye con causa registrada los SKU con huecos de calendario, demanda no
finita, historia insuficiente o sin observaciones reservadas, construye
`SelectionRequest`s tipados y ejecuta cada decisión del router como unidad
independiente (`SelectionRouter.plan` y `execute`). Every native strategy now
exposes `SelectionResult.forecast_config` in its model factory's format, plus
`forecast_seed`. Evidence remains in the original `payload`; no caller must
parse that evidence or reconstruct seasonal feature settings. Old custom
strategies remain usable with the router alone; a strategy used in coordinated
fitting must supply this new configuration boundary.

L2 devuelve, por candidato, la configuración seleccionada, la historia
admisible y el pronóstico de los días reservados desde t* (`FittedCandidate.forecast`).
El modelo ajustado no se conserva: se reconstruye con la fábrica, la
configuración y la semilla. L3 creates fresh models for each causal window; it
never evaluates using the full-history fitted instance. It preserves the existing walk-forward algorithm, metrics and
seed derivation. Configuration search and L3 evaluation use the same history:
`EvaluationArtifact.selection_scope` is `same_history`. These metrics are not an
unbiased holdout or a retrospective verdict. Statistical comparison remains
explicitly unavailable.

## Results and independent stages

A `PipelineResult` always represents exactly L1, L2, L3 and L4, with execution
states `pending`, `completed`, `blocked` or `failed`. Execution completion means
the configured working capability ran; repository maturity is a separate field.
Thus L3 can have a completed walk-forward execution while still being partial.
`complete` means all four configured stage operations returned artifacts, not
that all planned PRED scientific capabilities have been implemented.

With the repository's current default composition:

- L1, L2 and the implemented L3 capability complete without caller handoffs.
- `run.blocked_stage` is `L4`, `run.complete` is false, `run.validation` is None.
- `run.ingestion`, `run.fitting` and `run.evaluation` retain actual outputs.

Each operation is independently callable for diagnostics and integration:

```python
ingestion = pipeline.ingest(PipelineInput(csv_path="sales.csv", data_root="data"))
fitting = pipeline.fit(ingestion)
evaluation = pipeline.evaluate(fitting)
# pipeline.validate(evaluation) raises StageUnavailableError without real L4.
```

This manual sequence is for isolating stages, not required by `run`. The existing
`run_ingest`, `run_classify_csv`, router strategy `.select`, native `.fit` and
`evaluar_walk_forward` APIs also remain available.

Una unidad SKU × familia que falla en L2 o L3 no detiene su etapa: queda en
`FittingArtifact.units` o `EvaluationArtifact.units` con su error, y
`PipelineResult.failures` las reúne. Una etapa falla solo cuando ninguna unidad
produce resultado. Stage failures raise `PipelineExecutionError` with `.stage`,
`.result` and the original exception as `__cause__`. The partial result preserves completed-stage
outputs, marks the failing stage and leaves subsequent stages pending. A
missing L4 is instead an expected blocked result. Neither case fabricates a
successful continuation.

## Future attachment points

Applications with custom search spaces, budgets or existing recovery settings
can register their current strategies in `StrategyRegistry`, compose a
`SelectionRouter` and construct `Pipeline(router, evaluation, factories=...)`.
`pipeline_setup.model_factories()` provides the native family factories. These
contracts avoid coupling applications to the diagnostic CLI or its defaults.

`RetrospectiveValidator` is the explicit future L4 boundary:

```python
from pred_engine.pipeline import Pipeline

pipeline = Pipeline(
    router,
    evaluation_settings,
    factories=factories,
    validator=actual_retrospective_validator,
)
```

The validator consumes `EvaluationArtifact`, including original selection,
model and window evidence. It must return `ValidationArtifact` with exactly one
`SkuVerdict` per evaluated SKU and a nonempty `audit_bundle`. Verdicts are
`hold`, `partial` or `fail`. The adapter is marked `caller_provided`, never
misrepresented as a repository implementation. Completing this boundary needs
no changes to callers or the transition loop. No example validator is shipped.
L3 statistical comparison can be added to `EvaluationArtifact` and `evaluate`
without changing its input boundary or introducing another principal stage.

## Technical proof and diagnostics

After installation, run `python3 -m pred_engine.verify` (or
`uv run python -m pred_engine.verify`). The proof creates temporary canonical
input and exercises real bounded SARIMA HPO, model fitting and every causal
window. It checks the four-stage catalog, automatic transitions, exact L4
blockage, parity with native and independent stage entry points, and the
separate CLI diagnostic adapter. It enforces an elapsed-time budget of 30
seconds and requires no API key, foundation download or operator handoff.

Exit `0` means these checks passed, not that L4 exists. The JSON report keeps
`complete: false`, `blocked_stage: L4` and the explicit missing-stage message.
There are no simulated unfinished-stage outputs in this proof.

The existing operator CLI gains a utilitarian `run` diagnostic:

```bash
pred-engine run --csv sales.csv --data-root data --trials 4
pred-engine run --parquet data/processed/sales.parquet --families classical
```

It prints a JSON-safe stage summary and returns `7` on an unfinished-stage
blocker, `8` cuando además hubo unidades fallidas aisladas, `1` on failure, or
`0` only when all configured operations finish. La corrida queda persistida en
`{data_root}/runs/{run_id}/`; `--seed-csv` antepone la Fase 0 y `--workers`
fija la cantidad de procesos.
Existing `ingest`, `classify`, `probe`, `models` and `verify --parquet` behavior is
unchanged. This command is not the eventual human-facing UI.

`tests/test_pipeline.py` covers native parity, multiple SKUs, both CSV adapters,
classified-artifact entry, failure retention, core family configuration
handoffs, and the L4 callback contract. Its test-only L4 adapter proves boundary
wiring, not a scientific implementation, and is never used by the proof entry
point.
