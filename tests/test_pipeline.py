"""Application behavior, native-model parity and honest unfinished-stage boundaries."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pred_engine.cli import diagnose_pipeline, main
from pred_engine.comun.modelos import ClassifiedObservation
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda
from pred_engine.optimizacion.optimizadores.modelos_deep_learning.espacio_dl import (
    EspacioDL,
)
from pred_engine.optimizacion.optimizadores.modelos_deep_learning.estrategia import (
    DLSelectionStrategy,
)
from pred_engine.optimizacion.optimizadores.modelos_fundacionales.estrategia import (
    FoundationSelectionStrategy,
)
from pred_engine.optimizacion.optimizadores.modelos_machine_learning.espacio_ml import (
    EspacioML,
)
from pred_engine.optimizacion.optimizadores.modelos_machine_learning.estrategia import (
    MLSelectionStrategy,
    PresupuestoHPO,
)
from pred_engine.optimizacion.router import (
    TOPOLOGICAL_PROFILES,
    SelectionRequest,
    SelectionResult,
    SelectionRouter,
    StrategyRegistry,
)
from pred_engine.pipeline import (
    EvaluationSettings,
    Pipeline,
    PipelineExecutionError,
    PipelineInput,
    SkuVerdict,
    StageUnavailableError,
    ValidationArtifact,
)
from pred_engine.pipeline_setup import (
    ConfiguredTopologyPolicy,
    build_pipeline,
    model_factories,
)
from pred_engine.verify import (
    representative_csv,
    verification_pipeline,
    verify_pipeline,
)


def request_at(root: Path) -> PipelineInput:
    return PipelineInput(
        csv_path=representative_csv(root / "source.csv"),
        data_root=root / "data",
    )


def test_real_pipeline_proof_preserves_native_results_and_reports_l4(tmp_path):
    report = verify_pipeline(tmp_path)
    assert report["verification"] == "passed"
    assert report["complete"] is False
    assert report["blocked_stage"] == "L4"
    assert report["manual_handoffs"] == 0
    assert report["native_stage_parity"] == ["L1", "L2", "L3"]
    assert report["callers"] == ["library", "diagnostic_adapter"]
    assert report["elapsed_seconds"] < 30


def test_multi_sku_chronology_survives_shuffled_input(tmp_path):
    request = request_at(tmp_path)
    source = pd.read_csv(request.csv_path)
    second = source.assign(sku_id="second", demand_qty=source["demand_qty"] + 3)
    pd.concat([source, second]).sample(frac=1, random_state=7).to_csv(
        request.csv_path,
        index=False,
    )
    result = verification_pipeline().run(request)
    assert [c.selection.sku_id for c in result.fitting.candidates] == [
        "proof-sku",
        "second",
    ]
    np.testing.assert_array_equal(
        result.fitting.candidates[0].series[:3],
        [18.0, 19.35, 20.7],
    )
    np.testing.assert_array_equal(
        result.fitting.candidates[1].series[:3],
        [21.0, 22.35, 23.7],
    )
    assert [
        c.walk_forward.n_ventanas_evaluadas for c in result.evaluation.candidates
    ] == [4, 4]
    assert result.evaluation.selection_scope == "same_history"
    assert result.evaluation.statistical_comparison_available is False
    assert result.blocked_stage == "L4"


def test_existing_parquet_can_enter_without_repeating_csv_ingestion(tmp_path):
    pipeline = verification_pipeline()
    original = pipeline.run(request_at(tmp_path))
    resumed = pipeline.run(PipelineInput(parquet_path=original.ingestion.parquet_path))
    assert resumed.blocked_stage == "L4"
    np.testing.assert_array_equal(
        resumed.fitting.candidates[0].model.predict(2),
        original.fitting.candidates[0].model.predict(2),
    )
    assert resumed.evaluation.candidates[0].walk_forward.valor_agregado == (
        original.evaluation.candidates[0].walk_forward.valor_agregado
    )


def test_semantic_ingestion_adapter_keeps_provider_contract(tmp_path):
    class AcceptedProvider:
        def complete(self, prompt, *, temperature, timeout):
            return '{"status":"accepted","diagnostic":[]}'

    result = verification_pipeline().run(
        replace(request_at(tmp_path), provider=AcceptedProvider()),
    )
    assert result.ingestion.panel["sku_class"].unique().tolist() == ["smooth"]
    assert result.ingestion.panel["demand_qty"].iloc[0] == 18.0
    assert result.blocked_stage == "L4"
    assert (tmp_path / "data" / "raw" / "source.csv").is_file()


def test_l1_error_has_no_downstream_evidence(tmp_path):
    request = request_at(tmp_path)
    Path(request.csv_path).write_text("unknown,value\na,1\n")
    with pytest.raises(PipelineExecutionError) as raised:
        verification_pipeline().run(request)
    error = raised.value
    assert error.stage == "L1"
    assert [s.state for s in error.result.stages] == [
        "failed",
        "pending",
        "pending",
        "pending",
    ]
    assert error.result.complete is False
    assert error.__cause__ is not None
    code, diagnostic = diagnose_pipeline(verification_pipeline(), request)
    assert code == 1
    assert diagnostic["failed_stage"] == "L1"
    assert diagnostic["stages"][0]["state"] == "failed"


def test_l2_factory_error_preserves_real_ingestion(tmp_path):
    pipeline = verification_pipeline()
    pipeline.factories.clear()
    with pytest.raises(
        PipelineExecutionError, match="No model factory for classical"
    ) as raised:
        pipeline.run(request_at(tmp_path))
    assert raised.value.stage == "L2"
    assert raised.value.result.ingestion.panel["demand_qty"].iloc[0] == 18.0
    assert [s.state for s in raised.value.result.stages] == [
        "completed",
        "failed",
        "pending",
        "pending",
    ]


def test_l2_rejects_missing_daily_observation(tmp_path):
    pipeline = verification_pipeline()
    ingestion = pipeline.ingest(request_at(tmp_path))
    broken = replace(
        ingestion, panel=ingestion.panel.drop(index=3).reset_index(drop=True)
    )
    with pytest.raises(ValueError, match="daily grid"):
        pipeline.fit(broken)
    assert ingestion.panel["demand_qty"].iloc[3] == 22.05


def test_legacy_custom_strategy_must_supply_fitting_configuration(tmp_path):
    class LegacyStrategy:
        family = "classical"

        def select(self, request, profile):
            return SelectionResult(
                sku_id=request.sku_id,
                sku_class=request.sku_class,
                family=self.family,
                profile=profile,
                produced_by="legacy",
                payload={"evidence": "selection only"},
            )

    registry = StrategyRegistry()
    registry.register("classical", LegacyStrategy())
    pipeline = verification_pipeline()
    pipeline.router = SelectionRouter(
        ConfiguredTopologyPolicy(("classical",)), registry
    )
    with pytest.raises(PipelineExecutionError, match="forecast_config") as raised:
        pipeline.run(request_at(tmp_path))
    assert raised.value.stage == "L2"
    assert raised.value.result.ingestion.panel["demand_qty"].iloc[0] == 18.0


def test_l3_causal_window_error_preserves_fitting(tmp_path):
    pipeline = verification_pipeline()
    pipeline.settings = EvaluationSettings(min_train=100, metric="mae")
    with pytest.raises(PipelineExecutionError) as raised:
        pipeline.run(request_at(tmp_path))
    assert raised.value.stage == "L3"
    assert raised.value.result.fitting.candidates[0].selection.family == "classical"
    assert [s.state for s in raised.value.result.stages] == [
        "completed",
        "completed",
        "failed",
        "pending",
    ]


def test_l4_is_unavailable_not_a_fake_verdict(tmp_path):
    pipeline = verification_pipeline()
    result = pipeline.run(request_at(tmp_path))
    with pytest.raises(StageUnavailableError, match="not implemented"):
        pipeline.validate(result.evaluation)
    assert result.validation is None
    assert result.stages[3].message == "Verdicts and audit bundles are not implemented"
    assert result.blocked_stage == "L4"


def test_caller_supplied_l4_uses_existing_transition_path(tmp_path):
    # Test-only validator: proves the boundary, not PRED's absent scientific L4.
    def contract_validator(evaluation):
        candidate = evaluation.candidates[0]
        return ValidationArtifact(
            verdicts=(SkuVerdict(candidate.fitted.selection.sku_id, "partial"),),
            audit_bundle={"test_evidence": candidate.walk_forward.n_ventanas_evaluadas},
        )

    pipeline = verification_pipeline()
    pipeline.validator = contract_validator
    result = pipeline.run(request_at(tmp_path))
    assert [s.state for s in result.stages] == ["completed"] * 4
    assert result.complete is True
    assert result.stages[3].definition.maturity == "caller_provided"
    assert result.validation.verdicts[0].verdict == "partial"
    assert result.validation.audit_bundle == {"test_evidence": 4}


@pytest.mark.parametrize(
    "output",
    [
        None,
        ValidationArtifact((SkuVerdict("wrong-sku", "hold"),), {"evidence": 1}),
        ValidationArtifact((SkuVerdict("proof-sku", "hold"),), {}),
    ],
)
def test_invalid_l4_outputs_never_complete_a_run(tmp_path, output):
    pipeline = verification_pipeline()
    pipeline.validator = lambda evaluation: output
    with pytest.raises(PipelineExecutionError) as raised:
        pipeline.run(request_at(tmp_path))
    assert raised.value.stage == "L4"
    assert raised.value.result.complete is False
    assert (
        raised.value.result.evaluation.candidates[0].walk_forward.n_ventanas_evaluadas
        == 4
    )
    assert raised.value.result.stages[3].state == "failed"


def test_cli_reports_same_honest_blocker(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "pred_engine.pipeline_setup.build_pipeline",
        lambda *args, **kwargs: verification_pipeline(),
    )
    request = request_at(tmp_path)
    code = main(
        ["run", "--csv", str(request.csv_path), "--data-root", str(request.data_root)]
    )
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 7
    assert report["complete"] is False
    assert report["blocked_stage"] == "L4"
    assert [s["id"] for s in report["stages"]] == ["L1", "L2", "L3", "L4"]
    assert report["stages"][2]["maturity"] == "partial"


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"csv_path": "a.csv", "parquet_path": "a.parquet"},
        {"parquet_path": "a.parquet", "provider": object()},
    ],
)
def test_ambiguous_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PipelineInput(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_train": 0},
        {"min_train": 8, "horizon": 0},
        {"min_train": 8, "step": True},
        {"min_train": 8, "seasonality": 0},
        {"min_train": 1},
        {"min_train": 8, "metric": "unknown"},
        {"min_train": 8, "aggregation": "unknown"},
        {"min_train": 8, "trim": 0.5},
    ],
)
def test_invalid_evaluation_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        EvaluationSettings(**kwargs)


@pytest.mark.parametrize("families", [(), ("dl", "dl"), ("unknown",)])
def test_invalid_family_configuration_is_rejected(families):
    with pytest.raises(ValueError):
        ConfiguredTopologyPolicy(families)


def test_sparse_policy_does_not_enable_excluded_dl():
    decisions = ConfiguredTopologyPolicy(("classical", "dl")).decide("lumpy")
    assert [(d.family, d.profile) for d in decisions] == [
        ("classical", "sparse_variable")
    ]
    with pytest.raises(ValueError, match="n_trials"):
        build_pipeline(EvaluationSettings(40), n_trials=0)


@pytest.mark.parametrize("family", ["ml", "dl"])
def test_real_other_core_families_connect_to_fitting_and_walkforward(tmp_path, family):
    settings = EvaluationSettings(16, horizon=2, step=3, metric="mae", seasonality=1)
    rules = ReglasPoda(habilitar_poda_semantica=False)
    if family == "ml":
        strategy = MLSelectionStrategy(
            espacio=EspacioML(
                lags_min=3,
                lags_max=3,
                n_estimators_min=5,
                n_estimators_max=5,
                max_depth_min=2,
                max_depth_max=2,
                m=3,
            ),
            presupuestos={p: PresupuestoHPO(1, rules) for p in TOPOLOGICAL_PROFILES},
            min_train=16,
            horizonte=2,
            paso=3,
            metrica_objetivo="mae",
            seed=9,
        )
    else:
        strategy = DLSelectionStrategy(
            espacio=EspacioDL(
                lags=(3, 3),
                capas=(2, 2),
                unidades=(8, 8),
                epochs=(2, 2),
                batch_size=(8, 8),
            ),
            n_trials=1,
            min_train=16,
            horizonte=2,
            paso=3,
            metrica_objetivo="mae",
            estacionalidad=1,
            reglas=rules,
            seed=9,
        )
    registry = StrategyRegistry()
    registry.register(family, strategy)
    pipeline = Pipeline(
        SelectionRouter(ConfiguredTopologyPolicy((family,)), registry),
        settings,
        factories=model_factories(),
    )
    result = pipeline.run(request_at(tmp_path))
    candidate = result.fitting.candidates[0]
    assert candidate.selection.family == family
    assert candidate.selection.forecast_seed == 9
    assert result.evaluation.candidates[0].walk_forward.n_ventanas_evaluadas == 4
    assert result.blocked_stage == "L4"
    factory = model_factories()[family]
    native = factory(candidate.selection.forecast_config, seed=9).fit(candidate.series)
    np.testing.assert_array_equal(candidate.model.predict(2), native.predict(2))
    if family == "ml":
        # Seasonal features must survive the L2 boundary, not revert to m=1.
        assert candidate.model.estacionalidad == 3
        assert candidate.selection.forecast_config["m"] == 3


def test_optional_foundation_inference_failure_is_not_hidden(tmp_path, monkeypatch):
    from pred_engine.comun.modelos.modelos_fundacionales.errores import (
        ModeloFundacionalNoDisponibleError,
    )

    def missing_model(config):
        raise ModeloFundacionalNoDisponibleError("foundation extra unavailable")

    monkeypatch.setattr(
        "pred_engine.comun.modelos.modelos_fundacionales.chronos2.cargar_pipeline",
        missing_model,
    )
    pipeline = build_pipeline(
        EvaluationSettings(16, horizon=2, step=3, metric="mae"),
        families=("foundation",),
    )
    with pytest.raises(
        PipelineExecutionError, match="foundation extra unavailable"
    ) as raised:
        pipeline.run(request_at(tmp_path))
    assert raised.value.stage == "L3"
    assert raised.value.result.fitting.candidates[0].selection.forecast_config == {}
    assert raised.value.result.stages[3].state == "pending"
    assert raised.value.result.complete is False


def test_foundation_selection_exposes_frozen_factory_config_without_loading_weights():
    request = SelectionRequest(
        sku_id="F",
        sku_class="smooth",
        series=(
            ClassifiedObservation(
                sku_id="F",
                sku_class="smooth",
                timestamp=pd.Timestamp("2024-01-01"),
                demand_qty=2.0,
                lead_time_days=3,
            ),
        ),
    )
    selection = FoundationSelectionStrategy().select(request, "dense_stable")
    assert selection.forecast_config == {}
    model = model_factories()["foundation"](selection.forecast_config, seed=0)
    model.fit(np.array([1.0, 2.0]))
    assert model.configuracion.model_id == "amazon/chronos-2"
    assert selection.payload["optimizado"] is False
