import importlib
import hashlib
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


def probe_module():
    return importlib.import_module("figure8_temperature_probe")


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
    assert plan["preflight_temperatures"] == [
        temperature for _, temperature in module().SWEEP_TEMPERATURES
    ]
    assert plan["preflight_batches_per_rank"] == 3
    assert plan["require_completed_inactive_fresh_queue"] is True


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


def test_temperature_queue_binding_pins_every_control_source(tmp_path):
    m = module()
    plan, admission, reference = (
        tmp_path / "plan.json", tmp_path / "admission.json", tmp_path / "reference.json",
    )
    bundle = tmp_path / "bundle"
    for path in (plan, admission, reference):
        path.write_text(path.name)
    bundle.mkdir()
    (bundle / "completed.json").write_text("complete")
    runtime = SimpleNamespace(
        work=tmp_path / "fresh-t5-figure8-temperature-v1",
        bank_root=tmp_path / "durable",
        control=ROOT,
        scripts=SCRIPTS,
        gpu_indexes=(0, 1),
    )

    binding = m.queue_binding(runtime, plan, admission, bundle, reference)

    assert set(binding["source_sha256"]) == set(m.TEMPERATURE_CONTROL_FILES)
    assert binding["source_sha256"]["figure8_temperature.py"] == file_sha256(
        SCRIPTS / "figure8_temperature.py"
    )


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
            "status": "verified_durable_final", "selected_epoch": 12000,
            "final_epoch": 20000,
            "bundle": f"{name}-completed",
            "bundle_completed_sha256": "1" * 64,
            "bank_receipt_sha256": "2" * 64,
            "run_completed_sha256": "3" * 64,
            "index_sha256": "4" * 64,
            "codes_sha256": "5" * 64,
            "audit_sha256": "6" * 64,
            "spec_sha256": "7" * 64,
            "run_config_sha256": "8" * 64,
        }
        for name in m.TOKENIZER_ARMS
    }
    admission = {
        "status": "pass", "kind": "figure8_temperature_admission_v1",
        "temperatures": plan["temperatures"],
        "baseline_bundle": bundle.name,
        "baseline_bundle_completed_sha256": file_sha256(bundle / "completed.json"),
        "baseline": plan["baseline"],
        "baseline_results": plan["baseline_results"],
        "all_tokenizer_one_factor_arms_durable": True,
        "tokenizer_arms": arms,
        "existing_fresh_queue_unchanged": True,
        "existing_fresh_queue_cells": 39,
        "existing_fresh_queue_binding_sha256": "3" * 64,
        "fresh_queue_durably_complete": True,
        "fresh_queue_driver_inactive": True,
        "fresh_queue_result_acks": {
            f"run-{index}": "4" * 64 for index in range(39)
        },
        "source_sha256": {
            name: "9" * 64 for name in m.TEMPERATURE_CONTROL_FILES
        },
        "temperature_preflight": {
            "receipt_sha256": "5" * 64,
            "durable_bank_receipt_sha256": "8" * 64,
            "stage_sha256": "6" * 64,
            "audit_sha256": "7" * 64,
            "initial_model_identical": True,
            "ordered_batches_identical": True,
            "temperatures": [
                temperature for _, temperature in m.SWEEP_TEMPERATURES
            ],
            "batches_per_rank": 3,
        },
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


def test_temperature_admission_rejects_missing_queue_completion_or_preflight(tmp_path):
    m = module()
    plan = m.load_plan(ROOT / "configs" / "letter-figure8-temperature-v1.json")
    bundle = tmp_path / plan["baseline"]["bundle_name"]
    write_json(bundle / "completed.json", {"status": "complete"})
    admission = {
        "status": "pass", "kind": "figure8_temperature_admission_v1",
        "temperatures": plan["temperatures"],
        "baseline_bundle": bundle.name,
        "baseline_bundle_completed_sha256": file_sha256(bundle / "completed.json"),
        "baseline": plan["baseline"],
        "baseline_results": plan["baseline_results"],
        "all_tokenizer_one_factor_arms_durable": True,
        "tokenizer_arms": {
            name: {
                "status": "verified_durable_final", "selected_epoch": 12000,
                "final_epoch": 20000,
                "bundle": f"{name}-completed",
                "bundle_completed_sha256": "1" * 64,
                "bank_receipt_sha256": "2" * 64,
                "run_completed_sha256": "3" * 64,
                "index_sha256": "4" * 64,
                "codes_sha256": "5" * 64,
                "audit_sha256": "6" * 64,
                "spec_sha256": "7" * 64,
                "run_config_sha256": "8" * 64,
            }
            for name in m.TOKENIZER_ARMS
        },
        "existing_fresh_queue_unchanged": True,
        "existing_fresh_queue_cells": 39,
        "existing_fresh_queue_binding_sha256": "3" * 64,
        "fresh_queue_durably_complete": True,
        "fresh_queue_driver_inactive": True,
        "fresh_queue_result_acks": {
            f"run-{index}": "4" * 64 for index in range(39)
        },
        "source_sha256": {
            name: "9" * 64 for name in m.TEMPERATURE_CONTROL_FILES
        },
        "temperature_preflight": {
            "receipt_sha256": "5" * 64,
            "durable_bank_receipt_sha256": "8" * 64,
            "stage_sha256": "6" * 64,
            "audit_sha256": "7" * 64,
            "initial_model_identical": True,
            "ordered_batches_identical": True,
            "temperatures": [
                temperature for _, temperature in m.SWEEP_TEMPERATURES
            ],
            "batches_per_rank": 3,
        },
    }
    path = tmp_path / "admission.json"
    write_json(path, admission)
    del admission["temperature_preflight"]["durable_bank_receipt_sha256"]
    write_json(path, admission)

    with pytest.raises(ValueError, match="prerequisites"):
        m.validate_admission(path, file_sha256(path), plan, bundle)


def test_temperature_admission_requires_tokenizer_spec_proof(tmp_path):
    m = module()
    plan = m.load_plan(ROOT / "configs" / "letter-figure8-temperature-v1.json")
    bundle = tmp_path / plan["baseline"]["bundle_name"]
    write_json(bundle / "completed.json", {"status": "complete"})
    arms = {
        name: {
            "status": "verified_durable_final", "selected_epoch": 12000,
            "final_epoch": 20000,
            "bundle": f"{name}-completed",
            "bundle_completed_sha256": "1" * 64,
            "bank_receipt_sha256": "2" * 64,
            "run_completed_sha256": "3" * 64,
            "index_sha256": "4" * 64,
            "codes_sha256": "5" * 64,
            "audit_sha256": "6" * 64,
            "spec_sha256": "7" * 64,
            "run_config_sha256": "8" * 64,
        }
        for name in m.TOKENIZER_ARMS
    }
    admission = {
        "status": "pass", "kind": "figure8_temperature_admission_v1",
        "temperatures": plan["temperatures"],
        "baseline_bundle": bundle.name,
        "baseline_bundle_completed_sha256": file_sha256(bundle / "completed.json"),
        "baseline": plan["baseline"],
        "baseline_results": plan["baseline_results"],
        "all_tokenizer_one_factor_arms_durable": True,
        "tokenizer_arms": arms,
        "existing_fresh_queue_unchanged": True,
        "existing_fresh_queue_cells": 39,
        "existing_fresh_queue_binding_sha256": "3" * 64,
        "fresh_queue_durably_complete": True,
        "fresh_queue_driver_inactive": True,
        "fresh_queue_result_acks": {
            f"run-{index}": "4" * 64 for index in range(39)
        },
        "source_sha256": {
            name: "9" * 64 for name in m.TEMPERATURE_CONTROL_FILES
        },
        "temperature_preflight": {
            "receipt_sha256": "5" * 64,
            "durable_bank_receipt_sha256": "8" * 64,
            "stage_sha256": "6" * 64,
            "audit_sha256": "7" * 64,
            "initial_model_identical": True,
            "ordered_batches_identical": True,
            "temperatures": plan["preflight_temperatures"],
            "batches_per_rank": plan["preflight_batches_per_rank"],
        },
    }
    del arms[m.TOKENIZER_ARMS[0]]["spec_sha256"]
    path = tmp_path / "admission.json"
    write_json(path, admission)

    with pytest.raises(ValueError, match="prerequisites"):
        m.validate_admission(path, file_sha256(path), plan, bundle)


def test_admitted_preflight_must_bind_actual_reconstructed_stage_and_audit(tmp_path):
    m = module()
    stage = tmp_path / "stages" / "Instruments-full-ids-v1" / "stage.json"
    audit = tmp_path / "audits" / "Instruments-full-ids-v1" / "audit.json"
    stage.parent.mkdir(parents=True)
    audit.parent.mkdir(parents=True)
    stage.write_text("stage")
    audit.write_text("audit")
    admission = {
        "temperature_preflight": {
            "stage_sha256": file_sha256(stage),
            "audit_sha256": file_sha256(audit),
        },
    }

    m.verify_admitted_preflight(tmp_path, admission)

    stage.write_text("tampered")
    with pytest.raises(ValueError, match="stage/audit"):
        m.verify_admitted_preflight(tmp_path, admission)


def test_admission_sources_must_match_runtime_control():
    m = module()
    admission = {"source_sha256": m.control_source_hashes(SCRIPTS)}

    m.verify_admission_sources(SCRIPTS, admission)

    admission["source_sha256"]["fresh_pipeline.py"] = "f" * 64
    with pytest.raises(ValueError, match="control source"):
        m.verify_admission_sources(SCRIPTS, admission)


def preflight_receipt():
    probes = {}
    for label, temperature in module().SWEEP_TEMPERATURES:
        probes[label] = {
            "temperature": temperature,
            "ranks": {
                str(rank): {
                    "initial_model_sha256": "1" * 64,
                    "ordered_batch_sha256": [f"{rank + 3}" * 64, f"{rank + 5}" * 64],
                }
                for rank in range(2)
            },
        }
    return {
        "status": "pass",
        "kind": "figure8_temperature_preflight_v1",
        "temperatures": [temperature for _, temperature in module().SWEEP_TEMPERATURES],
        "world_size": 2,
        "batches_per_rank": 2,
        "stage_sha256": "a" * 64,
        "audit_sha256": "b" * 64,
        "probes": probes,
    }


def test_temperature_preflight_requires_identical_initialization_and_batch_order():
    m = module()
    receipt = preflight_receipt()

    assert m.validate_preflight(receipt)["status"] == "pass"

    receipt["probes"]["0p8"]["ranks"]["1"]["initial_model_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="initial model"):
        m.validate_preflight(receipt)


def test_temperature_preflight_rejects_order_drift():
    m = module()
    receipt = preflight_receipt()
    receipt["probes"]["1p2"]["ranks"]["0"]["ordered_batch_sha256"][0] = "f" * 64

    with pytest.raises(ValueError, match="batch order"):
        m.validate_preflight(receipt)


def test_temperature_preflight_rejects_rank_initialization_drift():
    m = module()
    receipt = preflight_receipt()
    receipt["probes"]["0p6"]["ranks"]["1"]["initial_model_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="ranks"):
        m.validate_preflight(receipt)


def test_temperature_preflight_assembles_exact_rank_probe_receipts(tmp_path):
    m = module()
    stage, audit = tmp_path / "stage.json", tmp_path / "audit.json"
    stage.write_text("stage")
    audit.write_text("audit")
    probe_root = tmp_path / "probes"
    for label, temperature in m.SWEEP_TEMPERATURES:
        directory = probe_root / label
        directory.mkdir(parents=True)
        for rank in range(2):
            write_json(directory / f"rank-{rank}.json", {
                "status": "pass",
                "kind": "figure8_temperature_probe_v1",
                "temperature": temperature,
                "rank": rank,
                "world_size": 2,
                "stage_sha256": file_sha256(stage),
                "audit_sha256": file_sha256(audit),
                "initial_model_sha256": "1" * 64,
                "ordered_batch_sha256": [
                    f"{rank + 3}" * 64, f"{rank + 5}" * 64,
                ],
            })

    receipt = m.assemble_preflight(probe_root, stage, audit, batches_per_rank=2)

    assert receipt["status"] == "pass"
    assert receipt["probes"]["1p0"]["temperature"] == 1.0
    assert receipt["stage_sha256"] == file_sha256(stage)


def test_temperature_probe_hashes_model_and_batches_canonically():
    torch = pytest.importorskip("torch")
    probe = probe_module()

    class Model:
        def state_dict(self):
            return {
                "z": torch.tensor([[1.0, 2.0]]),
                "a": torch.tensor([3], dtype=torch.int64),
            }

    model_hash = probe.model_state_sha256(Model())
    assert model_hash == probe.model_state_sha256(Model())
    assert probe.batch_sha256({
        "labels": torch.tensor([[1, 2]]),
        "input_ids": torch.tensor([[3, 4]]),
    }) == probe.batch_sha256({
        "input_ids": torch.tensor([[3, 4]]),
        "labels": torch.tensor([[1, 2]]),
    })


def write_sweep_arm(path: Path, temperature: float) -> None:
    source_run = {
        "dataset": "Instruments", "seed": 42, "temperature": temperature,
        "inputs": {"train": "1" * 64},
        "author_source": {"data.py": "2" * 64},
        "options": {"num_train_epochs": 200},
        "world_size": 2,
        "initialization": "author random T5 configuration",
        "tokenizer_vocab_sha256": "3" * 64,
        "model_config": {"d_model": 128},
        "harness_sha256": {"run.py": "4" * 64},
        "trainer_source_sha256": "5" * 64,
        "versions": {"torch": "1.13.1"},
        "experiment_lineage": {
            "index_sha256": "6" * 64, "codes_sha256": "7" * 64,
        },
    }
    metric_values = {
        "recall@1": 1.0, "recall@5": 1.0, "recall@10": 1.0,
        "ndcg@5": 1.0, "ndcg@10": 1.0,
    }
    rows = [
        {
            "user": str(index), "target_item": index, "target_code": f"<a_{index}>",
            "metrics": {"code": metric_values, "item": metric_values},
        }
        for index in range(2)
    ]
    path.mkdir()
    predictions = path / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(path / "evaluation-config.json", {
        "dataset": "Instruments", "split": "test", "beams": 20,
        "expected_users": 2, "source_run": source_run,
    })
    write_json(path / "completed.json", {
        "status": "complete", "users": 2,
        "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
        "metrics": {"code": metric_values, "item": metric_values},
    })


def test_temperature_sweep_aggregates_all_seven_points_and_six_fixed_contrasts(tmp_path):
    m = module()
    arms = {}
    for label, temperature in m.SWEEP_TEMPERATURES:
        arms[label] = tmp_path / label
        write_sweep_arm(arms[label], temperature)

    result = m.aggregate_sweep(arms, seed=20261027, replicates=20)

    assert [point["temperature"] for point in result["points"]] == [
        0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
    ]
    assert set(result["contrasts"]) == {"0p6", "0p7", "0p8", "0p9", "1p1", "1p2"}
    assert all(value["right_temperature"] == 1.0 for value in result["contrasts"].values())
    assert "best_temperature" not in result
    assert result["selection"] == "descriptive only; no test-based winner"


def test_temperature_sweep_rejects_missing_arm(tmp_path):
    m = module()
    arms = {}
    for label, temperature in m.SWEEP_TEMPERATURES[:-1]:
        arms[label] = tmp_path / label
        write_sweep_arm(arms[label], temperature)

    with pytest.raises(ValueError, match="exact seven-point"):
        m.aggregate_sweep(arms, seed=20261027, replicates=20)


def write_durable_tree(path: Path, files: dict[str, str]) -> None:
    manifest = {"files": {}}
    for name, content in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode())
        manifest["files"][name] = {
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "bytes": target.stat().st_size,
        }
    write_json(path / "bank-receipt.json", manifest)


def test_tokenizer_barrier_rehashes_every_durable_final_bundle(tmp_path):
    m = module()
    audits = {}
    durable = tmp_path / "durable"
    specs = tmp_path / "specs"
    specs.mkdir()
    for index, arm in enumerate(m.TOKENIZER_ARMS):
        bundle = f"{arm}-completed-{index:02d}"
        target = durable / bundle
        spec = {
            "dataset": "Instruments", "codes": [256, 256, 256, 256],
            "epsilons": [0.0, 0.0, 0.0, 0.003], "dimension": 32,
            "layers": [2048, 1024, 512, 256, 128, 64], "clusters": 10,
            "alpha": 0.02, "beta": 0.0001, "seed": 42, "epochs": 20000,
            "batch_size": 1024, "eval_step": 2000,
        }
        write_json(specs / f"{arm}.json", spec)
        write_durable_tree(target, {
            "completed.json": json.dumps({"status": "complete", "arm": arm}),
            "payload/completed.json": json.dumps({"status": "complete"}),
            "payload/run-config.json": json.dumps(spec),
            "payload/export/index.json": json.dumps({"0": ["<a_0>"]}),
            "payload/export/codes.npy": f"codes-{arm}",
        })
        audit = {
            "status": "pass", "arm": arm, "bundle": bundle,
            "selected_epoch": 12000, "final_epoch": 20000,
            "index_sha256": file_sha256(target / "payload/export/index.json"),
            "codes_sha256": file_sha256(target / "payload/export/codes.npy"),
            "bundle_completed_sha256": file_sha256(target / "completed.json"),
            "run_completed_sha256": file_sha256(target / "payload/completed.json"),
            "durable_bank_receipt_sha256": file_sha256(target / "bank-receipt.json"),
        }
        audits[arm] = tmp_path / f"{arm}.json"
        write_json(audits[arm], audit)

    result = m.verify_tokenizer_barrier(audits, durable, specs)

    assert set(result) == set(m.TOKENIZER_ARMS)
    assert all(value["status"] == "verified_durable_final" for value in result.values())

    tampered = durable / result[m.TOKENIZER_ARMS[0]]["bundle"] / "payload/export/codes.npy"
    tampered.write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        m.verify_tokenizer_barrier(audits, durable, specs)


def test_fresh_queue_barrier_requires_all_39_durable_results_and_active_binding(tmp_path):
    m = module()
    work = tmp_path / "fresh-t5-v3"
    durable = tmp_path / "durable"
    work.mkdir()
    cells = [
        {
            "dataset": "Instruments", "variant": "full", "seed": index,
            "temperature": 1.0, "rq_run_id": "rq", "stage_id": "stage",
            "run_id": f"run-{index:02d}",
        }
        for index in range(39)
    ]
    binding_path = work / "queue-binding-rq-compat-v6.json"
    write_json(binding_path, {
        "queue": cells, "durable_root": str(durable), "work": str(work),
    })
    write_json(work / "driver.json", {
        "queue_binding_file": binding_path.name,
        "queue_binding_sha256": file_sha256(binding_path),
    })
    write_json(work / "status.json", {"status": "all_39_cells_durably_complete"})
    for cell in cells:
        report = work / "results" / cell["run_id"]
        write_json(report / "result.json", {"run_id": cell["run_id"]})
        write_json(work / "acks" / f"{cell['run_id']}.json", {
            "status": "banked", "run_id": cell["run_id"],
            "result_sha256": file_sha256(report / "result.json"),
        })
        write_durable_tree(
            durable / "results" / cell["run_id"],
            {"result.json": (report / "result.json").read_text()},
        )

    result = m.verify_fresh_queue_artifacts(
        work, binding_path, cells, durable,
    )

    assert len(result) == 39
    assert set(result) == {cell["run_id"] for cell in cells}

    (work / "acks" / "run-38.json").unlink()
    with pytest.raises(ValueError, match="durably complete"):
        m.verify_fresh_queue_artifacts(work, binding_path, cells, durable)


def test_baseline_admission_requires_exact_rq_durable_and_result_hashes(tmp_path):
    m = module()
    plan = m.load_plan(ROOT / "configs" / "letter-figure8-temperature-v1.json")
    bundle = tmp_path / plan["baseline"]["bundle_name"]
    bundle.mkdir()
    (bundle / "completed.json").write_text("bundle")
    plan["baseline"]["bundle_completed_sha256"] = file_sha256(bundle / "completed.json")
    results = {
        "temperature_0p7_result_sha256": tmp_path / "t07.json",
        "temperature_1p0_result_sha256": tmp_path / "t10.json",
    }
    for field, path in results.items():
        path.write_text(field)
        plan["baseline_results"][field] = file_sha256(path)
    rq = {
        "rq_run_id": plan["baseline_rq_run_id"],
        "bundle_completed_sha256": plan["baseline"]["bundle_completed_sha256"],
        "run_completed_sha256": plan["baseline"]["run_completed_sha256"],
        "final_epoch": 20000,
        "final_state_sha256": plan["baseline"]["final_state_sha256"],
        "selected_epoch": plan["baseline"]["selected_epoch"],
        "selected_state_sha256": plan["baseline"]["selected_state_sha256"],
        "index_sha256": plan["baseline"]["index_sha256"],
        "codes_sha256": plan["baseline"]["codes_sha256"],
    }
    durable = {
        "bank_receipt_sha256": plan["baseline"]["durable_bank_receipt_sha256"],
        "bundle_completed_sha256": plan["baseline"]["bundle_completed_sha256"],
        "run_completed_sha256": plan["baseline"]["run_completed_sha256"],
    }

    assert m.verify_baseline(plan, bundle, rq, durable, results) == plan["baseline"]

    rq["selected_state_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="baseline"):
        m.verify_baseline(plan, bundle, rq, durable, results)


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
