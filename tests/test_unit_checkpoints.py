"""Reanudacion durable: no repetir exitos, reintentar fallos y conservar evidencia."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pred_engine.pipeline as module
from pred_engine.comun.ejecucion_paralela import ejecutar_unidades
from pred_engine.pipeline import PipelineExecutionError, PipelineInput
from pred_engine.unit_checkpoints import UnitCheckpoints, load_result, save_result
from pred_engine.verify import representative_csv, verification_pipeline


def _parallel_probe(task):
    folder, index = task
    if index == 0:
        deadline = time.monotonic() + 20
        while not (folder / "release").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("la primera unidad no fue liberada")
            time.sleep(0.02)
    return index


def _persist_probe(task, result):
    folder, index = task
    save_result(folder / f"{index}.json", result)


@pytest.mark.slow
def test_parallel_completion_is_durable_even_while_parent_waits_for_an_earlier_unit(
    tmp_path,
):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            ejecutar_unidades,
            _parallel_probe,
            [(tmp_path, 0), (tmp_path, 1)],
            procesos=2,
            al_completar=_persist_probe,
        )
        try:
            deadline = time.monotonic() + 15
            while not (tmp_path / "1.json").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            restored = load_result(Path(tmp_path / "1.json"))
            assert restored is not None and restored.valor == 1
            assert not future.done()
            assert not (tmp_path / "0.json").exists()
        finally:
            (tmp_path / "release").touch()
        assert [r.valor for r in future.result(timeout=20)] == [0, 1]


def setup_run(tmp_path, *, workers=1, identity="prueba-v1"):
    csv = representative_csv(tmp_path / "source.csv")
    frame = pd.read_csv(csv)
    pd.concat([frame, frame.assign(sku_id="second")]).to_csv(csv, index=False)
    pipeline = verification_pipeline()
    pipeline.workers = workers
    pipeline.progress = tmp_path / "progreso.jsonl"
    pipeline.checkpoints = UnitCheckpoints(tmp_path / "checkpoints", identity=identity)
    return pipeline, PipelineInput(csv_path=csv, data_root=tmp_path / "data")


@pytest.mark.parametrize("workers", [1, 2])
def test_completed_units_are_restored_without_calling_selection_fit_or_evaluation(
    tmp_path, monkeypatch, workers
):
    pipeline, request = setup_run(tmp_path, workers=workers)
    first = pipeline.run(request)
    snapshot = {p: p.read_bytes() for p in pipeline.checkpoints.root.rglob("*.json")}
    # Dos SKU por dos familias en L2 y en L3.
    assert len(snapshot) == 8

    def forbidden(*args):
        raise AssertionError("una unidad exitosa no debe volver a ejecutarse")

    monkeypatch.setattr(module, "_fit_unit", forbidden)
    monkeypatch.setattr(module, "_evaluate_unit", forbidden)
    restored = pipeline.run(request)
    assert all(u.reused for u in restored.units)
    assert {p: p.read_bytes() for p in snapshot} == snapshot
    for a, b in zip(first.fitting.candidates, restored.fitting.candidates, strict=True):
        assert a.selection == b.selection
        np.testing.assert_array_equal(a.forecast, b.forecast)
        np.testing.assert_array_equal(a.series, b.series)
    for a, b in zip(
        first.evaluation.candidates, restored.evaluation.candidates, strict=True
    ):
        assert a.walk_forward.valor_agregado == b.walk_forward.valor_agregado
        for x, y in zip(a.walk_forward.ventanas, b.walk_forward.ventanas, strict=True):
            assert x.metricas == y.metricas
            np.testing.assert_array_equal(x.y_pred, y.y_pred)
            np.testing.assert_array_equal(x.y_real, y.y_real)
            np.testing.assert_array_equal(x.y_train, y.y_train)
    assert [u.record for u in first.units] == [u.record for u in restored.units]
    events = [json.loads(line) for line in pipeline.progress.read_text().splitlines()]
    assert sum(e.get("reused", False) for e in events) == 8


@pytest.mark.parametrize("stage", ["L2", "L3"])
def test_interruption_retains_each_committed_unit_before_the_stage_finishes(
    tmp_path, monkeypatch, stage
):
    pipeline, request = setup_run(tmp_path)
    save = module.save_result
    counter = 0

    def interrupted(path, result):
        nonlocal counter
        if path.parent.name == stage:
            counter += 1
            if counter == 2:
                raise KeyboardInterrupt("cierre antes de confirmar la segunda unidad")
        save(path, result)

    monkeypatch.setattr(module, "save_result", interrupted)
    with pytest.raises(KeyboardInterrupt):
        pipeline.run(request)
    assert len(list((pipeline.checkpoints.root / stage).glob("*.json"))) == 1
    monkeypatch.setattr(module, "save_result", save)
    original = module._fit_unit if stage == "L2" else module._evaluate_unit
    called = []

    def record(task):
        called.append(
            (task.request.sku_id, task.decision.family)
            if stage == "L2"
            else (task.selection.sku_id, task.selection.family)
        )
        return original(task)

    monkeypatch.setattr(
        module, "_fit_unit" if stage == "L2" else "_evaluate_unit", record
    )
    resumed = pipeline.run(request)
    # Solo se repiten las tres unidades de la etapa que no llegaron a confirmarse.
    assert called == [("proof-sku", "ml"), ("second", "classical"), ("second", "ml")]
    assert len(resumed.evaluation.candidates) == 4
    assert sum(u.reused for u in resumed.units) == (1 if stage == "L2" else 5)


def test_retry_evaluates_only_the_failed_unit_and_keeps_the_successful_evidence(
    tmp_path, monkeypatch
):
    pipeline, request = setup_run(tmp_path)
    original = module._evaluate_unit

    def fail_one(task):
        if task.selection.sku_id == "second":
            raise ValueError("fallo transitorio")
        return original(task)

    monkeypatch.setattr(module, "_evaluate_unit", fail_one)
    first = pipeline.run(request)
    assert len(first.failures) == 2
    assert len(list((pipeline.checkpoints.root / "L3").glob("*.json"))) == 2
    called = []

    def record(task):
        called.append((task.selection.sku_id, task.selection.family))
        return original(task)

    monkeypatch.setattr(module, "_evaluate_unit", record)
    second = pipeline.run(request)
    assert not second.failures
    assert called == [("second", "classical"), ("second", "ml")]
    # Las cuatro unidades L2 y las dos L3 de proof-sku se reutilizan.
    assert sum(u.reused for u in second.units) == 6
    assert first.evaluation.units[0].record == second.evaluation.units[0].record


def test_changed_input_or_identity_never_reuses_incompatible_units(tmp_path):
    pipeline, request = setup_run(tmp_path)
    first = pipeline.run(request)
    assert not any(u.reused for u in first.units)
    pipeline.checkpoints = UnitCheckpoints(
        tmp_path / "checkpoints", identity="prueba-v2"
    )
    second = pipeline.run(request)
    assert not any(u.reused for u in second.units)
    frame = pd.read_csv(request.csv_path)
    frame["demand_qty"] += 2
    frame.to_csv(request.csv_path, index=False)
    request = replace(request, on_conflict="keep_both")
    third = pipeline.run(request)
    assert not any(u.reused for u in third.units)
    pipeline.settings = replace(pipeline.settings, horizon=3)
    fourth = pipeline.run(request)
    assert not any(u.reused for u in fourth.units)


def test_corrupt_checkpoint_is_rejected_without_overwriting_it(tmp_path):
    pipeline, request = setup_run(tmp_path)
    pipeline.run(request)
    path = next((pipeline.checkpoints.root / "L2").glob("*.json"))
    path.write_text('{"incompleto":true}')
    with pytest.raises(PipelineExecutionError, match="checkpoint invalido"):
        pipeline.run(request)
    assert path.read_text() == '{"incompleto":true}'


def test_persistence_failure_is_not_announced_as_success(tmp_path, monkeypatch):
    pipeline, request = setup_run(tmp_path)

    def cannot_store(*args):
        raise OSError("disco no disponible")

    monkeypatch.setattr(module, "save_result", cannot_store)
    with pytest.raises(PipelineExecutionError, match="disco no disponible"):
        pipeline.run(request)
    events = [json.loads(line) for line in pipeline.progress.read_text().splitlines()]
    finished = [e for e in events if e["event"] == "unit_finished"]
    assert finished and all(e["state"] == "fallida" for e in finished)
    assert not list(pipeline.checkpoints.root.rglob("*.json"))
