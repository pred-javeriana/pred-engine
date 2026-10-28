# DEPENDENCIES.md

Dependency manifest for `pred-engine` — updated on every dependency change
(SRS RNF traceability requirement).

## Runtime dependencies

| Package | Version constraint | Purpose |
|---------|-------------------|---------|
| numpy | >=1.26 | N-dimensional arrays; core data structure for all time-series operations |
| pandas | >=2.2 | Tabular I/O for passive CSV extraction and in-memory ingestion artefacts |
| pyarrow | >=16 | Columnar Parquet engine (`engine="pyarrow"`) for processed exports |
| pydantic | >=2.6 | Strict row/mapping contracts for semantic alignment (TASK-DATA-1.2) |
| httpx | >=0.27 | Stateless HTTP client with first-class timeouts for LLM providers |
| statsmodels | >=0.14 | SARIMAX (`comun.modelos.modelos_clasicos.sarima`) y descomposición STL (`aumentacion.mbb`) |
| optuna | >=4.0 | Sampler TPE multivariado y `Study`/pruner del motor de HPO (`optimizacion.optimizadores.HPO`) — ver ADR-02-006 |
| psutil | >=5.9 | CPU y memoria por proceso durante `pred-engine run` (`run_telemetry`, `recursos.jsonl`) |

## Extra `foundation`

`pred-engine run` corre la familia `foundation` por defecto, así que la
instalación del README incluye este extra:
`uv sync --frozen --extra dev --extra foundation`. Sin el extra, la corrida
necesita `--no-foundation`. CI no lo instala y su corrida de humo pasa
`--no-foundation`: las pruebas del modelo real llevan `importorskip` y
`@pytest.mark.slow`.

| Package | Version constraint | Purpose |
|---------|-------------------|---------|
| chronos-forecasting | ==2.3.2 | `Chronos2Pipeline` zero-shot (`comun.modelos.modelos_fundacionales`) — contrato en `docs/features/3.Evaluacion/3.2-adaptador-candidatos/README.md`, configuracion en `modelos_fundacionales/configuracion.py`. Debe coincidir con `CHRONOS2_ZERO_SHOT.version_libreria` |
| torch | >=2.2,<3 | Backend de inferencia de Chronos-2 (CPU, fp32, algoritmos deterministas) |

Pesos: `amazon/chronos-2` (Apache-2.0) en la revision fijada por
`CHRONOS2_ZERO_SHOT.revision`. La primera corrida los descarga una vez a la
cache de Hugging Face (`~/.cache/huggingface`, o `HF_HOME`) antes de L1; las
siguientes los leen de esa cache, tambien sin red con `HF_HUB_OFFLINE=1`.

> Direct runtime dependencies are added here as each layer is implemented.
> Vendor SDKs (google-genai, openai, anthropic) are intentionally not used;
> each provider is a thin `httpx` adapter behind `LlmProvider`.

## Development / test dependencies

| Package | Version constraint | Purpose |
|---------|-------------------|---------|
| pytest | >=8.0 | Test runner for the unit and integration suite |
| pytest-cov | >=5.0 | Coverage measurement; enforces the 80% gate on every CI run |
| ruff | >=0.5 | Linter and formatter (replaces flake8 + isort + black) |
| pre-commit | >=3.7 | Git hook runner; enforces ruff checks before every commit |
| pyright | >=1.1 | Comprobacion estatica de tipos en CI y desarrollo (`pyproject.toml`) |

## Lock file

Exact pinned versions for all transitive dependencies are recorded in
`uv.lock`, which is committed to the repository (RNF-REP-01: reproducible
execution from a clean clone).
