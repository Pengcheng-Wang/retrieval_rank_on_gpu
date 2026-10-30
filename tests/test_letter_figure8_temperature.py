import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))
from protocol import file_sha256, write_json


def module():
    return importlib.import_module("figure8_temperature")


def test_temperature_queue_is_exact_and_excludes_existing_baselines():
    cells = module().temperature_cells()

    assert [cell["temperature"] for cell in cells] == [0.6, 0.8, 0.9, 1.1, 1.2]
    assert all(cell["dataset"] == "Instruments" for cell in cells)
    assert all(cell["variant"] == "full" and cell["seed"] == 42 for cell in cells)
    assert all(cell["stage_id"] == "Instruments-full-ids-v1" for cell in cells)
    assert len({cell["run_id"] for cell in cells}) == 5
    assert not {0.7, 1.0} & {cell["temperature"] for cell in cells}


def test_frozen_temperature_plan_is_exact():
    plan = module().load_plan(ROOT / "configs" / "letter-figure8-temperature-v1.json")

    assert plan["tokenizer_barrier"] == list(module().TOKENIZER_ARMS)
    assert plan["baseline"]["selected_epoch"] == 18000
    assert plan["reference_fresh_queue_cells"] == 39


@pytest.mark.parametrize("field", ["temperatures", "tokenizer_barrier", "baseline", "dataset"])
def test_temperature_plan_rejects_scientific_drift(tmp_path, field):
    plan = json.loads((ROOT / "configs" / "letter-figure8-temperature-v1.json").read_text())
    plan[field] = [] if field != "dataset" else "Beauty"
    path = tmp_path / "plan.json"
    write_json(path, plan)

    with pytest.raises(ValueError, match="frozen scientific grid"):
        module().load_plan(path)


def test_temperature_queue_binding_is_immutable(tmp_path):
    binding = {"kind": "figure8_temperature_queue_v1", "temperatures": [0.6, 0.8]}

    marker = module().bind_temperature_queue(tmp_path, binding)

    assert json.loads(marker.read_text()) == binding
    assert module().bind_temperature_queue(tmp_path, binding) == marker
    with pytest.raises(ValueError, match="frozen"):
        module().bind_temperature_queue(tmp_path, {**binding, "temperatures": [0.6]})


def test_temperature_driver_runs_first_unbanked_arm_in_frozen_order(tmp_path, monkeypatch):
    m = module()
    runtime = SimpleNamespace(work=tmp_path, execute_cell=lambda cell, bundle: calls.append(cell))
    cells = m.temperature_cells()
    calls = []
    banked = {cells[0]["run_id"], cells[1]["run_id"]}
    monkeypatch.setattr(m, "cell_is_banked", lambda work, cell: cell["run_id"] in banked)

    result = m.execute_next(runtime, tmp_path / "bundle", cells)

    assert result["run_id"] == cells[2]["run_id"]
    assert calls == [cells[2]]


def test_temperature_driver_reports_complete_without_reexecuting(tmp_path, monkeypatch):
    m = module()
    cells = m.temperature_cells()
    runtime = SimpleNamespace(
        work=tmp_path,
        execute_cell=lambda *args: pytest.fail("Completed queue must not execute"),
    )
    monkeypatch.setattr(m, "cell_is_banked", lambda work, cell: True)

    assert m.execute_next(runtime, tmp_path / "bundle", cells) is None


def test_temperature_admission_requires_all_durable_tokenizer_arms(tmp_path):
    m = module()
    plan = m.load_plan(ROOT / "configs" / "letter-figure8-temperature-v1.json")
    bundle = tmp_path / plan["baseline"]["bundle_name"]
    write_json(bundle / "completed.json", {"status": "complete"})
    arms = {
        name: {
            "status": "verified_durable_final", "final_epoch": 20000,
            "bundle_completed_sha256": "1" * 64,
            "bank_receipt_sha256": "2" * 64,
        }
        for name in m.TOKENIZER_ARMS
    }
    admission = {
        "status": "pass", "kind": "figure8_temperature_admission_v1",
        "temperatures": plan["temperatures"],
        "baseline_bundle": bundle.name,
        "baseline_bundle_completed_sha256": file_sha256(bundle / "completed.json"),
        "baseline": plan["baseline"],
        "all_tokenizer_one_factor_arms_durable": True,
        "tokenizer_arms": arms,
        "existing_fresh_queue_unchanged": True,
        "existing_fresh_queue_cells": 39,
        "existing_fresh_queue_binding_sha256": "3" * 64,
        "gpu_indexes": [0, 1],
        "fresh_queue_gpu_indexes": [2, 3],
        "same_allocation_distinct_gpu_lease": True,
    }
    path = tmp_path / "admission.json"
    write_json(path, admission)

    assert m.validate_admission(path, file_sha256(path), plan, bundle) == admission
    del admission["tokenizer_arms"][m.TOKENIZER_ARMS[-1]]
    write_json(path, admission)
    with pytest.raises(ValueError, match="prerequisites"):
        m.validate_admission(path, file_sha256(path), plan, bundle)


@pytest.mark.parametrize(
    "value,expected",
    [("0,1", (0, 1)), ("6,7", (6, 7))],
)
def test_two_gpu_pair_parser(value, expected):
    assert module().parse_gpu_pair(value) == expected


@pytest.mark.parametrize("value", ["", "0", "0,0", "0,1,2", "-1,0", "a,b"])
def test_two_gpu_pair_parser_rejects_unsafe_values(value):
    with pytest.raises(ValueError):
        module().parse_gpu_pair(value)
