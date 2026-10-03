"""Offline integration proof: ``python -m pred_engine.verify``.

Exit 0 means the wiring checks passed, NOT that four-stage PRED is finished.
The proof uses real ingestion, bounded SARIMA HPO, fitting and walk-forward.
It never injects a pretend L4 validator or downloads foundation weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

from pred_engine.cli import diagnose_pipeline
from pred_engine.comun.modelos import ClassifiedObservation
from pred_engine.comun.walkforward import evaluar_walk_forward
from pred_engine.ingesta.pipeline import run_classify_csv, run_verify_parquet
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda
from pred_engine.optimizacion.optimizadores.modelos_clasicos import (
    EspacioClasico,
    fabrica_sarima,
)
from pred_engine.optimizacion.optimizadores.modelos_clasicos.estrategia import (
    ClassicalSelectionStrategy,
    PresupuestoClasico,
)
from pred_engine.optimizacion.router import (
    TOPOLOGICAL_PROFILES,
    SelectionRequest,
    SelectionRouter,
    StrategyRegistry,
)
from pred_engine.pipeline import (
    EvaluationSettings,
    Pipeline,
    PipelineInput,
    summarize_run,
)
from pred_engine.pipeline_setup import ConfiguredTopologyPolicy


def verification_pipeline() -> Pipeline:
    """Small real configuration; not a different orchestration implementation."""
    settings = EvaluationSettings(
        min_train=16,
        horizon=2,
        step=3,
        seasonality=1,
        metric="mae",
    )
    registry = StrategyRegistry()
    registry.register(
        "classical",
        ClassicalSelectionStrategy(
            espacio=EspacioClasico(
                p_max=1,
                d_max=0,
                q_max=0,
                P_max=0,
                D_max=0,
                Q_max=0,
                m=1,
            ),
            presupuestos={
                profile: PresupuestoClasico(
                    n_trials=1,
                    reglas=ReglasPoda(habilitar_poda_semantica=False),
                )
                for profile in TOPOLOGICAL_PROFILES
            },
            min_train=16,
            horizonte=2,
            paso=3,
            metrica_objetivo="mae",
            seed=17,
        ),
    )
    return Pipeline(
        SelectionRouter(ConfiguredTopologyPolicy(("classical",)), registry),
        settings,
        factories={"classical": fabrica_sarima},
    )


def representative_csv(path: Path) -> Path:
    """35 dias: 28 de historia admisible y 7 reservados para M3 (ADR-03-003)."""
    pd.DataFrame(
        {
            "sku_id": ["proof-sku"] * 35,
            "timestamp": pd.date_range("2024-01-01", periods=35),
            "demand_qty": [20 + 0.35 * i + (i % 5 - 2) for i in range(35)],
            "lead_time_days": [3] * 35,
        }
    ).to_csv(path, index=False)
    return path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_pipeline(root: Path) -> dict[str, object]:
    started = perf_counter()
    pipeline = verification_pipeline()
    csv = representative_csv(root / "representative.csv")
    request = PipelineInput(csv_path=csv, data_root=root / "integrated")
    result = pipeline.run(request)
    _require(
        [s.definition.id for s in result.stages] == ["L1", "L2", "L3", "L4"],
        "Expected exactly four principal stages",
    )
    _require(
        [s.state for s in result.stages]
        == ["completed", "completed", "completed", "blocked"],
        "Implemented stages must flow consecutively before the L4 blocker",
    )
    _require(
        not result.complete and result.blocked_stage == "L4",
        "Unfinished L4 must not be reported as a four-stage success",
    )
    ingestion, fitting, evaluation = result.ingestion, result.fitting, result.evaluation
    _require(
        ingestion is not None and fitting is not None and evaluation is not None,
        "Completed stages must provide real artifacts",
    )
    # Narrowing for static readers; the runtime requirement above works under -O.
    if ingestion is None or fitting is None or evaluation is None:
        raise RuntimeError("Missing evidence")
    _require(len(fitting.candidates) == 1, "The representative SKU must be fitted")
    _require(
        fitting.reserve.reserved_days == 7 and len(fitting.candidates[0].series) == 28,
        "L2 must only see the history before the 20% chronological reserve",
    )
    _require(
        evaluation.candidates[0].walk_forward.n_ventanas_evaluadas == 4,
        "Expected all four causal windows",
    )

    # Baseline entry points remain usable without orchestration.
    _, baseline_path = run_classify_csv(csv, data_root=root / "baseline")
    pd.testing.assert_frame_equal(ingestion.panel, run_verify_parquet(baseline_path))
    admissible = ingestion.panel.loc[
        ingestion.panel["timestamp"] <= fitting.reserve.t_star
    ]
    observations = tuple(
        ClassifiedObservation(**row) for row in admissible.to_dict(orient="records")
    )
    native_selection = pipeline.router.route(
        SelectionRequest(
            sku_id="proof-sku",
            sku_class=observations[0].sku_class,
            series=observations,
        )
    )[0]
    candidate = fitting.candidates[0]
    _require(
        candidate.selection == native_selection,
        "Orchestration changed native selection results",
    )
    config = native_selection.forecast_config
    _require(config is not None, "Selection must expose a factory-ready configuration")
    if config is None:
        raise RuntimeError("Missing configuration")
    native_model = fabrica_sarima(config, seed=17).fit(candidate.series)
    np.testing.assert_array_equal(candidate.forecast, native_model.predict(7))
    native_evaluation = evaluar_walk_forward(
        candidate.series,
        fabrica_sarima,
        config,
        min_train=16,
        horizonte=2,
        paso=3,
        metrica_objetivo="mae",
        estacionalidad=1,
        seed=17,
        identificador="proof-sku",
    )
    integrated = evaluation.candidates[0].walk_forward
    _require(
        integrated.valor_agregado == native_evaluation.valor_agregado,
        "Orchestration changed native walk-forward metrics",
    )
    for actual, expected in zip(
        integrated.ventanas, native_evaluation.ventanas, strict=True
    ):
        np.testing.assert_array_equal(actual.y_pred, expected.y_pred)
        np.testing.assert_array_equal(actual.y_real, expected.y_real)

    # Each stage is also callable independently through the application service.
    independent_ingestion = pipeline.ingest(request)
    pd.testing.assert_frame_equal(independent_ingestion.panel, ingestion.panel)
    independent_fit = pipeline.fit(independent_ingestion)
    np.testing.assert_array_equal(
        independent_fit.candidates[0].forecast, candidate.forecast
    )
    independent_evaluation = pipeline.evaluate(independent_fit)
    _require(
        independent_evaluation.candidates[0].walk_forward.valor_agregado
        == integrated.valor_agregado,
        "Independent L3 changed the result",
    )

    # A second caller (the CLI's diagnostic adapter) uses the same service loop.
    code, diagnostic = diagnose_pipeline(pipeline, request)
    _require(
        code == 7 and diagnostic == summarize_run(result),
        "Diagnostic caller must report the same L4 blocker",
    )
    elapsed = perf_counter() - started
    _require(elapsed < 30, "Verification exceeded the 30-second budget")
    return {
        "verification": "passed",
        **summarize_run(result),
        "manual_handoffs": 0,
        "callers": ["library", "diagnostic_adapter"],
        "native_stage_parity": ["L1", "L2", "L3"],
        "elapsed_seconds": round(elapsed, 3),
    }


def main() -> int:
    try:
        with TemporaryDirectory(prefix="pred-pipeline-proof-") as directory:
            report = verify_pipeline(Path(directory))
    except Exception as exc:
        print(
            json.dumps({"verification": "failed", "error": str(exc)}), file=sys.stderr
        )
        return 1
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
