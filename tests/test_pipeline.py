"""Application behavior, native-model parity and honest unfinished-stage boundaries."""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pred_engine.cli import diagnose_pipeline, main
from pred_engine.comun.errores_codigo import ELEGIBILIDAD_MENOS_DE_DOS_FAMILIAS
from pred_engine.comun.modelos import ClassifiedObservation
from pred_engine.comun.reserva import ReserveCut
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
    summarize_run,
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
from tests._dobles import PipelineFalso
from tests._plataforma import crear_base_plataforma
from tests._series import serie_ar_estacional


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
    assert [
        (c.selection.sku_id, c.selection.family) for c in result.fitting.candidates
    ] == [
        ("proof-sku", "classical"),
        ("proof-sku", "ml"),
        ("second", "classical"),
        ("second", "ml"),
    ]
    # Cada familia del SKU ve la misma historia ordenada.
    for candidate in result.fitting.candidates[:2]:
        np.testing.assert_array_equal(candidate.series[:3], [18.0, 19.35, 20.7])
    for candidate in result.fitting.candidates[2:]:
        np.testing.assert_array_equal(candidate.series[:3], [21.0, 22.35, 23.7])
    assert [
        c.walk_forward.n_ventanas_evaluadas for c in result.evaluation.candidates
    ] == [4, 4, 4, 4]
    assert result.evaluation.selection_scope == "same_history"
    assert result.evaluation.statistical_comparison_available is False
    assert result.blocked_stage == "L4"


def test_existing_parquet_can_enter_without_repeating_csv_ingestion(tmp_path):
    pipeline = verification_pipeline()
    original = pipeline.run(request_at(tmp_path))
    resumed = pipeline.run(PipelineInput(parquet_path=original.ingestion.parquet_path))
    assert resumed.blocked_stage == "L4"
    np.testing.assert_array_equal(
        resumed.fitting.candidates[0].forecast,
        original.fitting.candidates[0].forecast,
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


class _NullForecaster:
    def fit(self, y):
        return self

    def predict(self, horizon):
        return np.zeros(horizon)


def test_degenerate_final_forecast_fails_its_unit_under_rule_3(tmp_path):
    pipeline = verification_pipeline()
    pipeline.factories["classical"] = lambda config, *, seed: _NullForecaster()
    with pytest.raises(PipelineExecutionError, match="prediccion_nula") as raised:
        pipeline.run(request_at(tmp_path))
    assert raised.value.stage == "L2"
    assert "desde t*" in str(raised.value)
    assert "configuracion={'p': 1, 'd': 0, 'q': 0" in str(raised.value)


def test_degenerate_final_forecast_records_its_stable_code(tmp_path):
    pipeline = verification_pipeline()
    request = request_at(tmp_path)
    source = pd.read_csv(request.csv_path)
    second = source.assign(sku_id="second", demand_qty=source["demand_qty"] + 3)
    pd.concat([source, second]).to_csv(request.csv_path, index=False)
    sano = pipeline.factories["classical"]
    llamadas = []

    def fabrica(config, *, seed):
        llamadas.append(seed)
        if len(llamadas) == 1:
            return _NullForecaster()
        return sano(config, seed=seed)

    pipeline.factories["classical"] = fabrica
    fitting = pipeline.fit(pipeline.ingest(request))
    failed = [unit for unit in fitting.units if unit.failed]
    assert [unit.record.codigo for unit in failed] == ["model.degenerate_forecast"]
    assert len(fitting.candidates) == 1


def test_l2_rejects_missing_daily_observation(tmp_path):
    pipeline = verification_pipeline()
    ingestion = pipeline.ingest(request_at(tmp_path))
    broken = replace(
        ingestion, panel=ingestion.panel.drop(index=3).reset_index(drop=True)
    )
    with pytest.raises(ValueError, match="daily grid"):
        pipeline.fit(broken)
    assert ingestion.panel["demand_qty"].iloc[3] == 22.05


def _panel_with_admissible_days(pipeline, root, days):
    """Panel cuya historia admisible tiene `days` dias (reserva de ceil(20 %))."""
    ingestion = pipeline.ingest(request_at(root))
    total = next(
        n for n in range(days + 1, days * 2) if n - math.ceil(round(0.2 * n, 9)) == days
    )
    stamps = pd.date_range("2024-01-01", periods=total)
    panel = pd.DataFrame(
        {
            "sku_id": "proof-sku",
            "timestamp": stamps,
            "demand_qty": [20 + 0.35 * i + (i % 5 - 2) for i in range(len(stamps))],
            "lead_time_days": 3,
            "sku_class": ingestion.panel["sku_class"].iloc[0],
        }
    )[list(ingestion.panel.columns)]
    return replace(ingestion, panel=panel, reserve=ReserveCut.of(panel))


def test_l2_excludes_history_below_the_hpo_minimum_windows(tmp_path):
    pipeline = Pipeline(
        verification_pipeline().router,
        EvaluationSettings(min_train=40),
        factories={},
    )
    ingestion = _panel_with_admissible_days(pipeline, tmp_path, 60)
    with pytest.raises(ValueError, match="1 SKU excluidos") as raised:
        pipeline.fit(ingestion)
    assert str(raised.value).endswith(
        "proof-sku: historia admisible insuficiente: 60 dias, se requieren 68 "
        "(min_train + horizonte + 3 pasos)"
    )


@pytest.mark.parametrize(
    "families",
    [
        ("classical",),
        ("ml",),
        ("dl",),
        ("foundation",),
        ("classical", "ml", "dl", "foundation"),
    ],
)
def test_l2_minimum_history_excludes_the_sku_for_every_family(tmp_path, families):
    pipeline = build_pipeline(EvaluationSettings(min_train=40), families=families)
    ingestion = _panel_with_admissible_days(pipeline, tmp_path, 60)
    with pytest.raises(ValueError, match="60 dias, se requieren 68"):
        pipeline.fit(ingestion)


def test_l2_keeps_history_that_meets_the_hpo_minimum_windows(tmp_path):
    pipeline = verification_pipeline()  # 16 + 2 + 3 * 3 = 27 dias
    with pytest.raises(ValueError, match="26 dias, se requieren 27"):
        pipeline.fit(_panel_with_admissible_days(pipeline, tmp_path, 26))
    enough = pipeline.fit(_panel_with_admissible_days(pipeline, tmp_path, 27))
    assert not enough.exclusions and enough.candidates


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


def test_l3_causal_window_error_preserves_fitting(tmp_path, monkeypatch):
    def broken_walk_forward(*args, **kwargs):
        raise ValueError("ventana causal imposible")

    monkeypatch.setattr(
        "pred_engine.pipeline.evaluar_walk_forward", broken_walk_forward
    )
    pipeline = verification_pipeline()
    with pytest.raises(PipelineExecutionError, match="ventana causal") as raised:
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
    with pytest.raises(StageUnavailableError, match="base de plataforma"):
        pipeline.validate(result.evaluation)
    assert result.validation is None
    assert (
        result.stages[3].message
        == "M3 requiere una conexion a la base de plataforma para evaluar y publicar"
    )
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


# Con ruido: una serie determinista deja raices SARIMA en el borde (ADR-020).
CORE_DEMAND = np.round(serie_ar_estacional(120), 2)


def core_request_at(root: Path) -> PipelineInput:
    source = root / "sales.csv"
    n = len(CORE_DEMAND)
    pd.DataFrame(
        {
            "sku_id": ["core-sku"] * n,
            "timestamp": pd.date_range("2024-01-01", periods=n),
            "demand_qty": CORE_DEMAND,
            "lead_time_days": [3] * n,
        }
    ).to_csv(source, index=False)
    return PipelineInput(csv_path=source, data_root=root / "data")


def test_default_composition_fits_and_evaluates_all_four_families(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "pred_engine.comun.modelos.modelos_fundacionales.chronos2.cargar_pipeline",
        lambda config: PipelineFalso(),
    )
    pipeline = build_pipeline(
        EvaluationSettings(40, horizon=2, step=14, seasonality=7, metric="mae"),
        n_trials=1,
        seed=15,
    )
    result = pipeline.run(core_request_at(tmp_path))
    assert [c.selection.family for c in result.fitting.candidates] == [
        "classical",
        "ml",
        "dl",
        "foundation",
    ]
    assert [
        c.walk_forward.n_ventanas_evaluadas for c in result.evaluation.candidates
    ] == [4, 4, 4, 4]
    assert [s.state for s in result.stages] == [
        "completed",
        "completed",
        "completed",
        "blocked",
    ]
    assert result.validation is None
    assert result.complete is False
    assert result.blocked_stage == "L4"
    for candidate in result.evaluation.candidates:
        assert np.isfinite(candidate.walk_forward.valor_agregado)
        assert candidate.fitted.forecast.shape == (24,)
        assert np.all(np.isfinite(candidate.fitted.forecast))
        np.testing.assert_array_equal(
            candidate.walk_forward.ventanas[0].y_real, CORE_DEMAND[40:42]
        )


def test_real_cli_core_composition_publishes_m3(tmp_path, capsys):
    crear_base_plataforma(tmp_path / "data" / "pred.db")
    request = core_request_at(tmp_path)
    code = main(
        [
            "run",
            "--csv",
            str(request.csv_path),
            "--data-root",
            str(request.data_root),
            "--trials",
            "1",
            "--min-train",
            "40",
            "--horizon",
            "2",
            "--step",
            "14",
            "--metric",
            "mae",
            "--seed",
            "15",
            "--workers",
            "2",
            "--no-foundation",
        ]
    )
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0
    assert report["complete"] is True
    assert report["blocked_stage"] is None
    assert [s["state"] for s in report["stages"]] == ["completed"] * 4
    assert report["m3"]["estado"] == "PUBLICADA"
    assert report["m3"]["n_unidades"] == 92
    panel = pd.read_parquet(tmp_path / "data/processed/sales.parquet")
    assert panel.shape == (120, 5)
    np.testing.assert_array_equal(panel["demand_qty"].to_numpy(), CORE_DEMAND)


def test_cli_reports_same_honest_blocker(tmp_path, monkeypatch, capsys):
    crear_base_plataforma(tmp_path / "data" / "pred.db")
    monkeypatch.setattr(
        "pred_engine.pipeline_setup.build_pipeline",
        lambda *args, **kwargs: verification_pipeline(),
    )
    request = request_at(tmp_path)
    code = main(
        [
            "run",
            "--csv",
            str(request.csv_path),
            "--data-root",
            str(request.data_root),
            "--workers",
            "2",
            "--no-foundation",
        ]
    )
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 7
    assert report["complete"] is False
    assert report["blocked_stage"] == "L4"
    assert [s["id"] for s in report["stages"]] == ["L1", "L2", "L3", "L4"]
    assert report["stages"][2]["maturity"] == "partial"


def test_cli_runs_all_four_families_unless_foundation_is_dropped(
    tmp_path, monkeypatch, capsys
):
    crear_base_plataforma(tmp_path / "data" / "pred.db")
    families = []

    def capture(*args, **kwargs):
        families.append(kwargs["families"])
        return verification_pipeline()

    monkeypatch.setattr("pred_engine.pipeline_setup.build_pipeline", capture)
    monkeypatch.setattr(
        "pred_engine.comun.modelos.modelos_fundacionales.cargar_pipeline",
        lambda config: PipelineFalso(),
    )
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    args = [
        "run",
        "--csv",
        str(request_at(tmp_path).csv_path),
        "--data-root",
        str(tmp_path / "data"),
        "--workers",
        "1",
    ]
    assert main([*args, "--no-foundation"]) == 7
    assert "HF_HUB_OFFLINE" not in os.environ
    assert main(args) == 7
    # Chronos-2 cargo antes de la corrida: sus procesos leen la cache sin red.
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert families == [
        ("classical", "ml", "dl"),
        ("classical", "ml", "dl", "foundation"),
    ]
    with pytest.raises(SystemExit):
        main([*args, "--no-foundation", "--families", "ml"])


def test_cli_stops_before_the_run_when_chronos_cannot_load(
    tmp_path, monkeypatch, capsys
):
    from pred_engine.comun.modelos.modelos_fundacionales import (
        ModeloFundacionalNoDisponibleError,
    )

    def missing_model(config):
        raise ModeloFundacionalNoDisponibleError("pesos ausentes")

    monkeypatch.setattr(
        "pred_engine.comun.modelos.modelos_fundacionales.cargar_pipeline",
        missing_model,
    )
    monkeypatch.setattr(
        "pred_engine.pipeline_setup.build_pipeline",
        lambda *args, **kwargs: pytest.fail("the run must not start"),
    )
    code = main(
        [
            "run",
            "--csv",
            str(request_at(tmp_path).csv_path),
            "--data-root",
            str(tmp_path / "data"),
        ]
    )
    assert code == 1
    assert capsys.readouterr().err == (
        "error: pesos ausentes; para correr sin Chronos-2 use --no-foundation\n"
    )
    assert not (tmp_path / "data").exists()


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
        {"min_train": 1, "metric": "mase"},
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
    strategies = {
        "ml": MLSelectionStrategy(
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
        ),
        "dl": DLSelectionStrategy(
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
        ),
    }
    # Las dos familias juntas: con una sola, la regla de dos familias excluye el SKU.
    registry = StrategyRegistry()
    for name, strategy in strategies.items():
        registry.register(name, strategy)
    pipeline = Pipeline(
        SelectionRouter(ConfiguredTopologyPolicy(tuple(strategies)), registry),
        settings,
        factories=model_factories(),
    )
    result = pipeline.run(request_at(tmp_path))
    assert [c.selection.family for c in result.fitting.candidates] == ["ml", "dl"]
    (candidate,) = (
        c for c in result.fitting.candidates if c.selection.family == family
    )
    assert candidate.selection.forecast_seed == 9
    (evaluated,) = (
        c for c in result.evaluation.candidates if c.fitted.selection.family == family
    )
    assert evaluated.walk_forward.n_ventanas_evaluadas == 4
    assert result.blocked_stage == "L4"
    factory = model_factories()[family]
    native = factory(candidate.selection.forecast_config, seed=9).fit(candidate.series)
    np.testing.assert_array_equal(candidate.forecast, native.predict(7))
    if family == "ml":
        # Seasonal features must survive the L2 boundary, not revert to m=1.
        assert native.estacionalidad == 3
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
    # The forecast from t* is produced in L2, so the missing extra surfaces there.
    assert raised.value.stage == "L2"
    assert raised.value.result.ingestion.panel["demand_qty"].iloc[0] == 18.0
    assert [s.state for s in raised.value.result.stages[2:]] == ["pending", "pending"]
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


def test_reserve_holds_back_the_final_20_percent_of_the_calendar(tmp_path):
    result = verification_pipeline().run(request_at(tmp_path))
    reserve = result.fitting.reserve
    assert reserve.fraction == 0.2
    assert reserve.reserved_days == 7
    assert reserve.t_star == pd.Timestamp("2024-01-28")
    assert reserve.reserved_dates[0] == pd.Timestamp("2024-01-29")
    assert reserve.reserved_dates[-1] == pd.Timestamp("2024-02-04")
    candidate = result.fitting.candidates[0]
    assert len(candidate.series) == 28
    assert candidate.forecast.shape == (7,)
    assert summarize_run(result)["reserve"] == {
        "fraction": 0.2,
        "t_star": "2024-01-28",
        "first_reserved": "2024-01-29",
        "last_observed": "2024-02-04",
        "reserved_days": 7,
    }


def test_altering_reserved_observations_does_not_change_m2(tmp_path):
    pipeline = verification_pipeline()
    ingestion = pipeline.ingest(request_at(tmp_path))
    panel = ingestion.panel.copy()
    reserved = panel["timestamp"] > pd.Timestamp("2024-01-28")
    panel.loc[reserved, "demand_qty"] = panel.loc[reserved, "demand_qty"] * 50
    altered = replace(ingestion, panel=panel)

    original_fit, altered_fit = pipeline.fit(ingestion), pipeline.fit(altered)
    assert altered_fit.candidates[0].selection == original_fit.candidates[0].selection
    np.testing.assert_array_equal(
        altered_fit.candidates[0].forecast, original_fit.candidates[0].forecast
    )
    original_eval = pipeline.evaluate(original_fit).candidates[0].walk_forward
    altered_eval = pipeline.evaluate(altered_fit).candidates[0].walk_forward
    assert altered_eval.valor_agregado == original_eval.valor_agregado


class _RouterFailingForOneSku:
    """Real router, except that one SKU's selection raises inside its unit."""

    def __init__(self, router):
        self.router = router

    def plan(self, request):
        return self.router.plan(request)

    def execute(self, request, decision):
        if request.sku_id == "broken":
            raise RuntimeError("HPO interrumpido para este SKU")
        return self.router.execute(request, decision)


def test_failed_unit_is_isolated_and_short_sku_is_excluded_with_cause(tmp_path):
    request = request_at(tmp_path)
    source = pd.read_csv(request.csv_path)
    broken = source.assign(sku_id="broken")
    short = source.assign(sku_id="short").iloc[:10]
    pd.concat([source, broken, short]).to_csv(request.csv_path, index=False)
    pipeline = verification_pipeline()
    pipeline.router = _RouterFailingForOneSku(pipeline.router)

    result = pipeline.run(request)

    assert [s.state for s in result.stages] == [
        "completed",
        "completed",
        "completed",
        "blocked",
    ]
    assert [c.selection.sku_id for c in result.fitting.candidates] == [
        "proof-sku",
        "proof-sku",
    ]
    assert [(u.sku_id, u.family, u.failed) for u in result.fitting.units] == [
        ("broken", "classical", True),
        ("broken", "ml", True),
        ("proof-sku", "classical", False),
        ("proof-sku", "ml", False),
    ]
    assert result.stages[1].message == (
        "2 de 4 unidades fallaron (aisladas); 1 SKU excluidos con causa"
    )
    summary = summarize_run(result)
    assert summary["failures"] == [
        {
            "stage": "L2",
            "sku_id": "broken",
            "family": family,
            "error": "RuntimeError: HPO interrumpido para este SKU",
            "code": "pipeline.stage_failed",
        }
        for family in ("classical", "ml")
    ]
    assert summary["exclusions"] == [
        {
            "sku_id": "short",
            "cause": "la serie termina antes de t*: no tiene observaciones reservadas",
            "code": "eligibility.no_reserved_observations",
        }
    ]
    assert summary["units"] == {
        "L2": {"total": 4, "completed": 2, "failed": 2},
        "L3": {"total": 2, "completed": 2, "failed": 0},
    }


def test_parallel_workers_reproduce_sequential_results_in_other_processes(tmp_path):
    request = core_request_at(tmp_path)
    source = pd.read_csv(request.csv_path)
    second = source.assign(sku_id="core-b", demand_qty=source["demand_qty"] * 1.5)
    pd.concat([source, second]).to_csv(request.csv_path, index=False)
    settings = EvaluationSettings(40, horizon=2, step=14, seasonality=7, metric="mae")

    def run(workers):
        # Con esta semilla SARIMA ajusta los dos SKU: con una sola familia
        # evaluable saldrian con la regla de dos familias (ADR-023).
        pipeline = build_pipeline(
            settings, families=("classical", "ml"), n_trials=1, seed=20, workers=workers
        )
        return pipeline.run(request)

    sequential, parallel = run(1), run(2)
    assert {
        (c.fitted.selection.sku_id, c.fitted.selection.family)
        for c in sequential.evaluation.candidates
    } == {(s, f) for s in ("core-sku", "core-b") for f in ("classical", "ml")}
    pairs = zip(
        sequential.evaluation.candidates, parallel.evaluation.candidates, strict=True
    )
    for one, other in pairs:
        assert one.fitted.selection == other.fitted.selection
        np.testing.assert_array_equal(one.fitted.forecast, other.fitted.forecast)
        assert one.walk_forward.valor_agregado == other.walk_forward.valor_agregado
    assert {u.record.pid for u in sequential.units} == {os.getpid()}
    parallel_pids = {u.record.pid for u in parallel.units}
    assert os.getpid() not in parallel_pids
    assert len(parallel_pids) >= 2
    assert summarize_run(parallel) == summarize_run(sequential)


def test_hpo_studies_resume_from_their_manifests_without_retraining(tmp_path):
    request = core_request_at(tmp_path)
    settings = EvaluationSettings(40, horizon=2, step=14, seasonality=7, metric="mae")
    hpo = tmp_path / "hpo"

    def run():
        return build_pipeline(
            settings,
            families=("classical", "ml"),
            n_trials=2,
            seed=3,
            hpo_root=hpo,
            session="s1",
        ).run(request)

    first = run()
    manifests = sorted(hpo.glob("*/manifiesto.json"))
    estudios = ["classical-core-sku-s1", "ml-core-sku-s1"]
    assert [m.parent.name for m in manifests] == estudios
    assert all(json.loads(m.read_text())["estado"] == "completada" for m in manifests)
    assert [
        c.selection.payload["estudio_hpo"] for c in first.fitting.candidates
    ] == estudios
    snapshot = {path: path.read_bytes() for path in hpo.rglob("*") if path.is_file()}

    second = run()
    for again, original in zip(
        second.fitting.candidates, first.fitting.candidates, strict=True
    ):
        assert again.selection == original.selection
        np.testing.assert_array_equal(again.forecast, original.forecast)
    assert {p: p.read_bytes() for p in hpo.rglob("*") if p.is_file()} == snapshot


def test_hpo_persistence_requires_root_and_session_together():
    with pytest.raises(ValueError, match="together"):
        build_pipeline(EvaluationSettings(40), hpo_root="runs")


def _seed_run_args(tmp_path: Path) -> list[str]:
    from tests.aumentacion.test_fase0 import _semilla_transaccional

    return [
        "run",
        "--seed-csv",
        str(_semilla_transaccional(tmp_path)),
        "--data-root",
        str(tmp_path / "data"),
        "--m0-metodo",
        "mbb-directo",
        "--m0-n-series",
        "1",
        "--m0-minimo-filas",
        "0",
        "--m0-columna",
        "sku_id=Item_ID",
        "--m0-columna",
        "timestamp=Date",
        "--m0-columna",
        "demand_qty=Avg_Usage_Per_Day",
        "--m0-columna",
        "lead_time_days=Restock_Lead_Time",
        "--families",
        "classical",
        "ml",
        "--trials",
        "2",
        "--min-train",
        "60",
        "--step",
        "28",
        "--workers",
        "2",
        # Con esta semilla unos SKU entregan las dos familias y otros solo
        # LightGBM, y estos salen con la regla de dos familias (ADR-023).
        "--seed",
        "16",
    ]


def test_cli_runs_m0_m1_m2_from_a_seed_and_persists_a_resumable_run(tmp_path, capsys):
    from pred_engine.comun.modelos.manifiesto_candidatos import ContextoParticion
    from pred_engine.forecasting.adaptador_candidatos import (
        instanciar,
        validar_manifiesto,
    )

    crear_base_plataforma(tmp_path / "data" / "pred.db")
    args = _seed_run_args(tmp_path)
    code = main(args)
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    assert code == (8 if report["failures"] or report["m3"]["n_fallos"] else 0)
    assert [s["state"] for s in report["stages"]] == ["completed"] * 4
    assert report["m3"]["estado"] == "PUBLICADA"
    run_dir = Path(report["run_dir"])
    assert run_dir.parent == tmp_path / "data" / "runs"
    run = json.loads((run_dir / "corrida.json").read_text())
    assert run["inputs"]["m0"]["rows"] == run["inputs"]["m1"]["rows"]
    assert run["inputs"]["m1"]["skus"] == 6
    assert run["inputs"]["source"]["sha256"] == run["inputs"]["m0"]["sha256"]
    assert run["summary"]["reserve"]["reserved_days"] == 48
    assert run["parallelism"]["L2"]["procesos_distintos"] >= 2

    # M3 accepts every candidate against the partition it derives on its own.
    panel = pd.read_parquet(run["inputs"]["m1"]["parquet"])
    manifest = (run_dir / "candidatos.json").read_text()
    handoff = validar_manifiesto(
        manifest,
        esperado=ContextoParticion(
            ingesta_ref_m1=run["inputs"]["m1"]["sha256"],
            t_corte_reserva=date.fromisoformat(run["summary"]["reserve"]["t_star"]),
            fraccion_reserva=0.2,
        ),
        skus_panel=set(panel["sku_id"]),
    )
    assert handoff.run_id_m2 == report["run_id"]
    assert handoff.fallos == ()
    candidates = {(c.sku, c.familia) for c in handoff.candidatos}
    skus = {sku for sku, _ in candidates}
    assert candidates == {(sku, f) for sku in skus for f in ("classical", "ml")}
    # Cada SKU excluido entrego una sola unidad L2 y no llega a M3.
    excluded = {e["sku_id"]: e["code"] for e in report["exclusions"]}
    assert skus and excluded and skus.isdisjoint(excluded)
    assert set(excluded.values()) == {ELEGIBILIDAD_MENOS_DE_DOS_FAMILIAS}
    assert len(candidates) == report["units"]["L2"]["completed"] - len(excluded)
    for candidate in handoff.candidatos:
        instanciar(candidate)
    assert [c["candidato_id"] for c in run["candidates"]] == [
        c.candidato_id for c in handoff.candidatos
    ]
    assert all(c["estudio_hpo"] for c in run["candidates"])
    assert run["versions"]["pred-engine"] == run["settings"]["pred_engine"]
    forecasts = pd.read_parquet(run_dir / "pronosticos.parquet")
    assert len(forecasts) == 48 * len(candidates)
    assert (forecasts["pronostico"] >= 0).all()
    assert forecasts["timestamp"].min() == pd.Timestamp(
        run["summary"]["reserve"]["first_reserved"]
    )
    evaluation = pd.read_parquet(run_dir / "evaluacion.parquet")
    assert len(evaluation) == report["units"]["L3"]["completed"]
    units = [json.loads(line) for line in (run_dir / "unidades.jsonl").open()]
    assert len(units) == report["units"]["L2"]["total"] + report["units"]["L3"]["total"]
    assert {u["model"] for u in units} <= {"sarima", "lightgbm"}
    assert all(u["cpu_s"] > 0 for u in units if u["state"] == "completada")
    assert set(run["stage_times"]) == {"L1", "L2", "L3", "L4"}
    assert run["stage_times"]["L2"]["seconds"] > 0

    # Resources over time, per worker process, beside the unit trace.
    assert "recursos.jsonl" in report["files"]
    assert run["files"]["recursos"] == "recursos.jsonl"
    assert run["telemetry"]["samples"] > 0 and run["telemetry"]["error"] is None
    samples = [json.loads(line) for line in (run_dir / "recursos.jsonl").open()]
    worker_pids = {s["pid"] for s in samples if s["role"] == "worker"}
    # A pool that lives less than one interval may fall between two samples.
    assert worker_pids & {u["pid"] for u in units}
    hpo = {p: p.read_bytes() for p in (run_dir / "hpo").rglob("*") if p.is_file()}
    assert hpo

    # Same command, same identity: M0 is reused and no HPO study is re-run.
    assert main(args) == code
    again = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert again["run_id"] == report["run_id"]
    rerun = json.loads((run_dir / "corrida.json").read_text())
    assert rerun["inputs"]["m0"]["reused"] is True
    assert {p: p.read_bytes() for p in hpo} == hpo
    first, rebuilt = (
        json.loads(manifest),
        json.loads((run_dir / "candidatos.json").read_text()),
    )
    assert rebuilt["candidatos"] == first["candidatos"]
    assert rebuilt["contexto"] == first["contexto"]
    assert rebuilt["emitido_en"] == first["emitido_en"]


def test_cli_reports_isolated_unit_failures_with_exit_code_8(
    tmp_path, monkeypatch, capsys
):
    crear_base_plataforma(tmp_path / "data" / "pred.db")

    def failing_pipeline(*args, **kwargs):
        pipeline = verification_pipeline()
        pipeline.router = _RouterFailingForOneSku(pipeline.router)
        return pipeline

    monkeypatch.setattr("pred_engine.pipeline_setup.build_pipeline", failing_pipeline)
    request = request_at(tmp_path)
    source = pd.read_csv(request.csv_path)
    pd.concat([source, source.assign(sku_id="broken")]).to_csv(
        request.csv_path, index=False
    )
    code = main(
        [
            "run",
            "--csv",
            str(request.csv_path),
            "--data-root",
            str(request.data_root),
            "--workers",
            "2",
            "--no-foundation",
        ]
    )
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 8
    assert report["blocked_stage"] == "L4"
    assert [(f["sku_id"], f["stage"], f["family"]) for f in report["failures"]] == [
        ("broken", "L2", "classical"),
        ("broken", "L2", "ml"),
    ]
    units = [
        json.loads(line) for line in (Path(report["run_dir"]) / "unidades.jsonl").open()
    ]
    assert sorted(
        (u["sku_id"], u["stage"], u["family"], u["state"]) for u in units
    ) == [
        ("broken", "L2", "classical", "fallida"),
        ("broken", "L2", "ml", "fallida"),
        ("proof-sku", "L2", "classical", "completada"),
        ("proof-sku", "L2", "ml", "completada"),
        ("proof-sku", "L3", "classical", "completada"),
        ("proof-sku", "L3", "ml", "completada"),
    ]


def test_cli_resource_sampling_can_be_disabled(tmp_path, monkeypatch, capsys):
    crear_base_plataforma(tmp_path / "data" / "pred.db")
    monkeypatch.setattr(
        "pred_engine.pipeline_setup.build_pipeline",
        lambda *args, **kwargs: verification_pipeline(),
    )
    request = request_at(tmp_path)
    args = [
        "run",
        "--csv",
        str(request.csv_path),
        "--data-root",
        str(request.data_root),
        "--workers",
        "2",
        "--no-foundation",
    ]

    assert main([*args, "--telemetry-interval", "-1"]) == 1
    assert "--telemetry-interval" in capsys.readouterr().err

    assert main([*args, "--telemetry-interval", "0"]) == 7
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    run_dir = Path(report["run_dir"])
    assert "recursos.jsonl" not in report["files"]
    assert not (run_dir / "recursos.jsonl").exists()
    assert json.loads((run_dir / "corrida.json").read_text())["telemetry"] is None


def test_cli_seed_errors_are_reported_without_traceback(tmp_path, capsys):
    crear_base_plataforma(tmp_path / "data" / "pred.db")
    code = main(
        [
            "run",
            "--seed-csv",
            str(tmp_path / "missing.csv"),
            "--data-root",
            str(tmp_path / "data"),
            "--no-foundation",
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error: ")
    assert "Traceback" not in captured.err
