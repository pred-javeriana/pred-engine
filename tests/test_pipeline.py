"""Application behavior, native-model parity and honest unfinished-stage boundaries."""

from __future__ import annotations

import gc
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pred_engine import pipeline as pipeline_module
from pred_engine.cli import diagnose_pipeline, main
from pred_engine.comun.dataclasses.validacion_temporal import ResultadoWalkForward
from pred_engine.comun.errores_codigo import (
    ELEGIBILIDAD_MENOS_DE_DOS_FAMILIAS,
    codigos_conocidos,
)
from pred_engine.comun.modelos import ClassifiedObservation
from pred_engine.comun.reserva import ReserveCut
from pred_engine.comun.walkforward import evaluar_walk_forward
from pred_engine.optimizacion.optimizadores.HPO.poda import ReglasPoda
from pred_engine.optimizacion.optimizadores.modelos_clasicos.classical_selection import (  # noqa: E501
    EspacioClasico,
    fabrica_sarima,
)
from pred_engine.optimizacion.optimizadores.modelos_clasicos.estrategia import (
    ClassicalSelectionStrategy,
    PresupuestoClasico,
)
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
from pred_engine.unit_checkpoints import UnitCheckpoints
from pred_engine.verify import (
    representative_csv,
    verification_pipeline,
    verify_pipeline,
)
from tests._dobles import PipelineFalso
from tests._plataforma import crear_base_plataforma
from tests._series import serie_ar_estacional

# La unidad L2 original, antes de que un test la reemplace con monkeypatch.
_L2_REAL = pipeline_module._fit_unit


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


class _ErrorConDosArgumentos(Exception):
    def __init__(self, codigo, detalle):
        super().__init__(f"{codigo}: {detalle}")


class _FalloConErrorDeConstructorExigente:
    def fit(self, y):
        raise _ErrorConDosArgumentos("E1", "ajuste roto")


def test_final_fit_error_with_required_args_keeps_the_real_error(tmp_path):
    pipeline = verification_pipeline()
    pipeline.factories["classical"] = lambda config, *, seed: (
        _FalloConErrorDeConstructorExigente()
    )
    with pytest.raises(PipelineExecutionError, match="E1: ajuste roto") as raised:
        pipeline.run(request_at(tmp_path))
    assert raised.value.stage == "L2"
    assert "TypeError" not in str(raised.value)


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


def test_l2_history_exclusion_reports_required_and_available_days(tmp_path):
    progress = tmp_path / "progreso.jsonl"
    pipeline = Pipeline(
        verification_pipeline().router,
        EvaluationSettings(min_train=40),
        factories={},
        progress=progress,
    )
    ingestion = _panel_with_admissible_days(pipeline, tmp_path, 60)
    with pytest.raises(ValueError, match="1 SKU excluidos"):
        pipeline.fit(ingestion)
    [excluded] = [
        json.loads(line)
        for line in progress.read_text().splitlines()
        if json.loads(line)["event"] == "excluded"
    ]
    assert excluded["code"] == "eligibility.insufficient_history"
    assert excluded["facts"] == {"history": {"required": 68, "available": 60}}


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

    # La unidad L3 entera: SARIMA ya no recalcula su walk-forward (usa el del HPO).
    monkeypatch.setattr("pred_engine.pipeline._evaluate_unit", broken_walk_forward)
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


@pytest.mark.parametrize("phase", ["fit", "predict"])
def test_final_fit_failure_names_configuration_and_exception(tmp_path, phase):
    class BrokenModel:
        def fit(self, y):
            if phase == "fit":
                raise RuntimeError("ajuste final interrumpido")
            return self

        def predict(self, horizon):
            raise RuntimeError("prediccion final interrumpida")

    pipeline = verification_pipeline()
    pipeline.factories = {
        family: lambda config, *, seed: BrokenModel() for family in ("classical", "ml")
    }
    with pytest.raises(PipelineExecutionError) as raised:
        pipeline.run(request_at(tmp_path))
    assert raised.value.stage == "L2"
    message = str(raised.value)
    assert "RuntimeError:" in message
    assert "configuracion=" in message
    assert "'lags': 3" in message  # configuracion ML elegida, no el espacio


@pytest.mark.parametrize("with_healthy", [False, True])
def test_unrouted_lumpy_is_excluded_without_losing_other_skus(tmp_path, with_healthy):
    request = core_request_at(tmp_path)
    source = pd.read_csv(request.csv_path)
    lumpy = source.assign(
        sku_id="lumpy-only", demand_qty=np.resize([0.0, 1.0, 0.0, 100.0], len(source))
    )
    pd.concat([source, lumpy] if with_healthy else [lumpy]).to_csv(
        request.csv_path, index=False
    )
    pipeline = build_pipeline(
        EvaluationSettings(40, horizon=2, step=14, seasonality=7, metric="mae"),
        families=("ml", "dl"),
        n_trials=1,
        seed=15,
    )
    ingestion = pipeline.ingest(request)
    if not with_healthy:
        with pytest.raises(ValueError, match="lumpy-only:.*clase 'lumpy'"):
            pipeline.fit(ingestion)
        return
    fitting = pipeline.fit(ingestion)
    assert [(c.selection.sku_id, c.selection.family) for c in fitting.candidates] == [
        ("core-sku", "ml"),
        ("core-sku", "dl"),
    ]
    assert all(unit.sku_id == "core-sku" for unit in fitting.units)
    assert len(fitting.exclusions) == 1
    exclusion = fitting.exclusions[0]
    assert exclusion.sku_id == "lumpy-only"
    assert exclusion.code == "eligibility.no_configured_family"
    assert exclusion.code in codigos_conocidos()
    assert "no asigna ninguna familia configurada a la clase 'lumpy'" in exclusion.cause
    assert pipeline.router.policy.version in exclusion.cause


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
    # Con el pool, el walk-forward del HPO viaja con el L3 y el candidato no
    # guarda otra copia de sus ventanas.
    assert all(c.fitted.walk_forward is None for c in parallel.evaluation.candidates)
    assert {u.record.pid for u in sequential.units} == {os.getpid()}
    parallel_pids = {u.record.pid for u in parallel.units}
    assert os.getpid() not in parallel_pids
    assert len(parallel_pids) >= 2
    assert summarize_run(parallel) == summarize_run(sequential)


def test_una_interrupcion_no_espera_las_unidades_que_no_empezaron(
    tmp_path, monkeypatch
):
    request = core_request_at(tmp_path)
    source = pd.read_csv(request.csv_path)
    pd.concat([source.assign(sku_id=f"core-{i}") for i in range(6)]).to_csv(
        request.csv_path, index=False
    )
    progreso = tmp_path / "progreso.jsonl"

    def interrumpir(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("pred_engine.pipeline.wait", interrumpir)
    pipeline = build_pipeline(
        EvaluationSettings(40, horizon=2, step=14, seasonality=7, metric="mae"),
        families=("classical", "ml"),
        n_trials=1,
        seed=3,
        workers=2,
        progress=progreso,
    )
    with pytest.raises(KeyboardInterrupt):
        pipeline.run(request)

    eventos = _eventos(progreso)
    [plan] = [e for e in eventos if (e["event"], e.get("stage")) == ("planned", "L2")]
    iniciadas = [e for e in eventos if e["event"] == "unit_started"]
    # Solo corren las unidades que ya estaban en manos del pool.
    assert len(iniciadas) < len(plan["units"]) == 12


def _clasicos(candidatos):
    """Los candidatos de la familia clasica (SARIMA) de una corrida."""
    return [c for c in candidatos if c.selection.family == "classical"]


def _l2_retenido(task):
    """L2 real; el de ``core-sku`` espera la senal del test (de modulo para spawn)."""
    if task.request.sku_id == "core-sku":
        senal = Path(os.environ["PRED_TEST_SENAL_L2"])
        limite = time.monotonic() + 120
        while not senal.exists():
            if time.monotonic() > limite:
                raise TimeoutError("el L2 retenido no fue liberado")
            time.sleep(0.02)
    return _L2_REAL(task)


def _eventos(progreso: Path) -> list[dict]:
    """Lineas completas de ``progreso.jsonl`` (la ultima puede estar escribiendose)."""
    texto = progreso.read_text(encoding="utf-8") if progreso.exists() else ""
    return [json.loads(linea) for linea in texto.split("\n")[:-1]]


def _ocurrio(progreso: Path, evento: str, etapa: str, sku: str) -> bool:
    return any(
        (e["event"], e.get("stage"), e.get("sku_id")) == (evento, etapa, sku)
        for e in _eventos(progreso)
    )


def _correr_con_l2_retenido(tmp_path, monkeypatch, hasta, **opciones):
    """Dos SKU y dos familias; los L2 de core-sku esperan a que ocurra ``hasta``.

    Con 3 procesos, las dos unidades retenidas dejan uno libre para core-b. Devuelve
    el resultado de la corrida y su flujo de progreso completo.
    """
    request = core_request_at(tmp_path)
    source = pd.read_csv(request.csv_path)
    second = source.assign(sku_id="core-b", demand_qty=source["demand_qty"] * 1.5)
    pd.concat([source, second]).to_csv(request.csv_path, index=False)
    senal = tmp_path / "soltar-l2"
    monkeypatch.setenv("PRED_TEST_SENAL_L2", str(senal))
    monkeypatch.setattr("pred_engine.pipeline._fit_unit", _l2_retenido)
    progreso = tmp_path / "progreso.jsonl"
    pipeline = build_pipeline(
        EvaluationSettings(40, horizon=2, step=14, seasonality=7, metric="mae"),
        families=("ml", "dl"),
        n_trials=1,
        seed=3,
        workers=3,
        progress=progreso,
        **opciones,
    )
    with ThreadPoolExecutor(max_workers=1) as hilo:
        corrida = hilo.submit(pipeline.run, request)
        try:
            limite = time.monotonic() + 60
            while not _ocurrio(progreso, *hasta):
                assert time.monotonic() < limite, f"{hasta} espero a todo L2"
                time.sleep(0.05)
            assert not _ocurrio(progreso, "unit_finished", "L2", "core-sku")
        finally:
            senal.touch()
        return corrida.result(timeout=120), _eventos(progreso)


def test_el_l3_de_una_unidad_empieza_antes_de_que_termine_todo_l2(
    tmp_path, monkeypatch
):
    result, _ = _correr_con_l2_retenido(
        tmp_path, monkeypatch, ("unit_started", "L3", "core-b")
    )
    assert result.evaluation is not None and len(result.evaluation.candidates) == 4


def test_con_l2_y_l3_solapados_cada_etapa_conserva_su_plan_y_sus_unidades(
    tmp_path, monkeypatch
):
    # Con checkpoints, el L3 adelantado ya guardo el suyo cuando evaluate lo busca.
    _, eventos = _correr_con_l2_retenido(
        tmp_path,
        monkeypatch,
        ("unit_finished", "L3", "core-b"),
        checkpoints=UnitCheckpoints(tmp_path / "checkpoints", identity="prueba-v1"),
    )
    skus = ["core-b", "core-b", "core-sku", "core-sku"]  # dos familias por SKU
    for etapa in ("L2", "L3"):
        [plan] = [
            e for e in eventos if (e["event"], e.get("stage")) == ("planned", etapa)
        ]
        assert sorted(u["sku_id"] for u in plan["units"]) == skus
        for evento in ("unit_started", "unit_finished"):
            unidades = [
                e["sku_id"]
                for e in eventos
                if (e["event"], e.get("stage")) == (evento, etapa)
            ]
            assert sorted(unidades) == skus, (evento, etapa)
    assert not any(e.get("reused") for e in eventos)
    etapas = [
        (e["event"], e["stage"])
        for e in eventos
        if e["event"] in ("stage_started", "stage_finished")
        and e["stage"] in ("L2", "L3")
    ]
    assert etapas == [
        ("stage_started", "L2"),
        ("stage_finished", "L2"),
        ("stage_started", "L3"),
        ("stage_finished", "L3"),
    ]


def test_l3_de_sarima_usa_el_walk_forward_del_hpo_sin_volver_a_ajustar(
    tmp_path, monkeypatch
):
    settings = EvaluationSettings(40, horizon=2, step=14, seasonality=7, metric="mae")
    pipeline = build_pipeline(
        settings, families=("classical", "ml"), n_trials=2, seed=3
    )
    fitting = pipeline.fit(pipeline.ingest(core_request_at(tmp_path)))
    [candidato] = _clasicos(fitting.candidates)
    # Lo que calcula L3 hoy para la configuracion ganadora.
    esperado = evaluar_walk_forward(
        candidato.series,
        pipeline.factories["classical"],
        dict(candidato.selection.forecast_config or {}),
        min_train=settings.min_train,
        horizonte=settings.horizon,
        paso=settings.step,
        metrica_objetivo=settings.metric,
        estacionalidad=settings.seasonality,
        agregacion=settings.aggregation,
        proporcion_recorte=settings.trim,
        seed=candidato.selection.forecast_seed,
        tolerar_fallos=False,
        identificador=candidato.selection.sku_id,
    )

    original = pipeline_module.evaluar_walk_forward

    def recalcular(*args, **kwargs):
        if args[1] is pipeline.factories["classical"]:
            raise AssertionError("L3 de SARIMA volvio a calcular el walk-forward")
        return original(*args, **kwargs)

    monkeypatch.setattr("pred_engine.pipeline.evaluar_walk_forward", recalcular)
    [evaluado] = [
        c
        for c in pipeline.evaluate(fitting).candidates
        if c.fitted.selection.family == "classical"
    ]

    assert evaluado.walk_forward.valor_agregado == esperado.valor_agregado
    for obtenida, calculada in zip(
        evaluado.walk_forward.ventanas, esperado.ventanas, strict=True
    ):
        np.testing.assert_array_equal(obtenida.y_pred, calculada.y_pred)
        assert dict(obtenida.metricas) == dict(calculada.metricas)


_AJUSTES_CORE = EvaluationSettings(40, horizon=2, step=14, seasonality=7, metric="mae")


def _router_clasico_ml(clase=SelectionRouter, estrategia=ClassicalSelectionStrategy):
    """SARIMA (un AR(1) sin estacionalidad, que siempre ajusta) y LightGBM."""
    comun = {
        "min_train": _AJUSTES_CORE.min_train,
        "horizonte": _AJUSTES_CORE.horizon,
        "paso": _AJUSTES_CORE.step,
        "metrica_objetivo": _AJUSTES_CORE.metric,
        "seed": 3,
    }
    registro = StrategyRegistry()
    registro.register(
        "classical",
        estrategia(
            **comun,
            espacio=EspacioClasico(
                p_max=1, d_max=0, q_max=0, P_max=0, D_max=0, Q_max=0
            ),
            presupuestos={
                p: PresupuestoClasico(n_trials=2) for p in TOPOLOGICAL_PROFILES
            },
        ),
    )
    registro.register(
        "ml",
        MLSelectionStrategy(
            **comun,
            presupuestos={p: PresupuestoHPO(n_trials=1) for p in TOPOLOGICAL_PROFILES},
        ),
    )
    return clase(ConfiguredTopologyPolicy(("classical", "ml")), registro)


class _Constante:
    def fit(self, y):
        return self

    def predict(self, horizon):
        return np.full(horizon, 10.0)


def _fabrica_constante(configuracion, *, seed):
    return _Constante()


def test_l3_evalua_la_fabrica_inyectada_y_no_el_walk_forward_del_hpo(tmp_path):
    # El HPO evalua SARIMA; L3 debe medir la fabrica clasica que recibe Pipeline.
    pipeline = Pipeline(
        _router_clasico_ml(),
        _AJUSTES_CORE,
        factories={**model_factories(), "classical": _fabrica_constante},
    )
    resultado = pipeline.run(core_request_at(tmp_path))

    [evaluado] = [
        c
        for c in resultado.evaluation.candidates
        if c.fitted.selection.family == "classical"
    ]
    for ventana in evaluado.walk_forward.ventanas:
        np.testing.assert_array_equal(ventana.y_pred, np.full(2, 10.0))


@wraps(fabrica_sarima)
def _sarima_envuelta(configuracion, *, seed):
    return _Constante()


def test_l3_mide_igual_una_fabrica_envuelta_al_correr_y_al_reanudar_l2(tmp_path):
    # Una fabrica envuelta conserva el nombre de fabrica_sarima pero es otro modelo:
    # L3 la mide a ella, con el L2 recien ajustado y con el L2 restaurado.
    pipeline = Pipeline(
        _router_clasico_ml(),
        _AJUSTES_CORE,
        factories={**model_factories(), "classical": _sarima_envuelta},
        checkpoints=UnitCheckpoints(tmp_path / "checkpoints", identity="prueba-v1"),
    )
    ingestion = pipeline.ingest(core_request_at(tmp_path))

    def l3_clasico(fitting):
        return next(
            c.walk_forward
            for c in pipeline.evaluate(fitting).candidates
            if c.fitted.selection.family == "classical"
        )

    recien = l3_clasico(pipeline.fit(ingestion))
    for checkpoint in (tmp_path / "checkpoints" / "L3").glob("*.json"):
        checkpoint.unlink()
    restaurado_l2 = pipeline.fit(ingestion)
    assert all(u.reused for u in restaurado_l2.units)
    reanudado = l3_clasico(restaurado_l2)

    for ventana in recien.ventanas:
        np.testing.assert_array_equal(ventana.y_pred, np.full(2, 10.0))
    assert recien.valor_agregado == reanudado.valor_agregado


def test_un_router_que_redefine_execute_sigue_eligiendo_en_l2(tmp_path):
    class RouterVigilado(SelectionRouter):
        llamadas = 0

        def execute(self, request, decision):
            RouterVigilado.llamadas += 1
            return super().execute(request, decision)

    pipeline = Pipeline(
        _router_clasico_ml(RouterVigilado),
        _AJUSTES_CORE,
        factories=model_factories(),
    )
    fitting = pipeline.fit(pipeline.ingest(core_request_at(tmp_path)))

    assert RouterVigilado.llamadas == 2  # una por familia del SKU
    assert all(c.walk_forward is None for c in fitting.candidates)


def test_una_estrategia_que_redefine_select_sigue_eligiendo_en_l2(tmp_path):
    class ClasicaVigilada(ClassicalSelectionStrategy):
        llamadas = 0

        def select(self, request, profile):
            ClasicaVigilada.llamadas += 1
            return super().select(request, profile)

    pipeline = Pipeline(
        _router_clasico_ml(estrategia=ClasicaVigilada),
        _AJUSTES_CORE,
        factories=model_factories(),
    )
    fitting = pipeline.fit(pipeline.ingest(core_request_at(tmp_path)))

    assert ClasicaVigilada.llamadas == 1
    assert all(c.walk_forward is None for c in fitting.candidates)


def _walk_forwards_del_hpo() -> int:
    """Walk-forwards del HPO vivos en este proceso (solo esos llevan ajustes)."""
    gc.collect()
    return sum(
        isinstance(o, ResultadoWalkForward) and o.ajustes is not None
        for o in gc.get_objects()
    )


def test_el_proceso_principal_suelta_el_walk_forward_del_hpo_apenas_sale_su_l3(
    tmp_path, monkeypatch
):
    request = core_request_at(tmp_path)
    source = pd.read_csv(request.csv_path)
    second = source.assign(sku_id="core-b", demand_qty=source["demand_qty"] * 1.5)
    pd.concat([source, second]).to_csv(request.csv_path, index=False)
    senal = tmp_path / "soltar-l2"
    monkeypatch.setenv("PRED_TEST_SENAL_L2", str(senal))
    monkeypatch.setattr("pred_engine.pipeline._fit_unit", _l2_retenido)
    progreso = tmp_path / "progreso.jsonl"
    pipeline = Pipeline(
        _router_clasico_ml(),
        _AJUSTES_CORE,
        factories=model_factories(),
        workers=3,
        progress=progreso,
    )
    with ThreadPoolExecutor(max_workers=1) as hilo:
        corrida = hilo.submit(pipeline.run, request)
        try:
            limite = time.monotonic() + 60
            while not _ocurrio(progreso, "unit_finished", "L3", "core-b"):
                assert time.monotonic() < limite, "el L3 de core-b no termino"
                time.sleep(0.05)
            # core-sku sigue en L2; el L3 SARIMA de core-b ya volvio al padre.
            [futuro, _] = pipeline._early_l3[("core-b", "classical")]
            assert futuro.result(timeout=30).valor.ajustes is not None
            assert not _ocurrio(progreso, "unit_finished", "L2", "core-sku")
            vivos = _walk_forwards_del_hpo()
        finally:
            senal.touch()
        corrida.result(timeout=120)
    # Solo el resultado de L3; L2 ya no guarda su copia.
    assert vivos == 1


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


def test_un_hpo_reanudado_sin_walk_forward_calcula_l3_como_hoy(tmp_path, monkeypatch):
    request = core_request_at(tmp_path)
    settings = EvaluationSettings(40, horizon=2, step=14, seasonality=7, metric="mae")

    def run():
        return build_pipeline(
            settings,
            families=("classical", "ml"),
            n_trials=2,
            seed=3,
            hpo_root=tmp_path / "hpo",
            session="s1",
        ).run(request)

    primera = run()  # L3 usa el walk-forward que calculo el HPO
    calculos = []
    original = pipeline_module.evaluar_walk_forward

    def contar(*args, **kwargs):
        if args[1] is model_factories()["classical"]:
            calculos.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr("pred_engine.pipeline.evaluar_walk_forward", contar)
    segunda = run()  # el estudio se rearma desde su manifiesto, sin walk-forward

    assert calculos == [1]
    [antes, despues] = [
        next(
            c
            for c in corrida.evaluation.candidates
            if c.fitted.selection.family == "classical"
        )
        for corrida in (primera, segunda)
    ]
    assert despues.walk_forward.valor_agregado == antes.walk_forward.valor_agregado
    for nueva, previa in zip(
        despues.walk_forward.ventanas, antes.walk_forward.ventanas, strict=True
    ):
        np.testing.assert_array_equal(nueva.y_pred, previa.y_pred)
        assert dict(nueva.metricas) == dict(previa.metricas)


def test_l3_no_reutiliza_un_walk_forward_calculado_con_otros_ajustes(
    tmp_path, monkeypatch
):
    # L3 agrega con la mediana; el HPO de SARIMA, con su media recortada.
    settings = EvaluationSettings(
        40, horizon=2, step=14, seasonality=7, metric="mae", aggregation="mediana"
    )
    pipeline = build_pipeline(
        settings, families=("classical", "ml"), n_trials=2, seed=3
    )
    fitting = pipeline.fit(pipeline.ingest(core_request_at(tmp_path)))
    [candidato] = _clasicos(fitting.candidates)
    esperado = evaluar_walk_forward(
        candidato.series,
        pipeline.factories["classical"],
        dict(candidato.selection.forecast_config or {}),
        min_train=settings.min_train,
        horizonte=settings.horizon,
        paso=settings.step,
        metrica_objetivo=settings.metric,
        estacionalidad=settings.seasonality,
        agregacion=settings.aggregation,
        proporcion_recorte=settings.trim,
        seed=candidato.selection.forecast_seed,
        tolerar_fallos=False,
        identificador=candidato.selection.sku_id,
    )
    calculos = []
    original = pipeline_module.evaluar_walk_forward

    def contar(*args, **kwargs):
        if args[1] is pipeline.factories["classical"]:
            calculos.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr("pred_engine.pipeline.evaluar_walk_forward", contar)
    [evaluado] = [
        c
        for c in pipeline.evaluate(fitting).candidates
        if c.fitted.selection.family == "classical"
    ]

    assert calculos == [1]
    assert evaluado.walk_forward.valor_agregado == esperado.valor_agregado


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
    # Un SKU excluido no corre L3, tampoco el que sale apenas cierra su L2.
    progress = [json.loads(line) for line in (run_dir / "progreso.jsonl").open()]
    assert not [
        e for e in progress if e.get("stage") == "L3" and e.get("sku_id") in excluded
    ]
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
    # El L3 de SARIMA reusa el walk-forward del HPO: no vuelve a ajustar ni gasta CPU.
    assert all(
        u["cpu_s"] > 0
        for u in units
        if u["state"] == "completada"
        and (u["stage"], u["family"]) != ("L3", "classical")
    )
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
