import copy
import importlib
import json
import os
from pathlib import Path
import socket
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))


def lifecycle():
    assert importlib.util.find_spec("lc_lifecycle"), "LC final-run lifecycle is not implemented"
    return importlib.import_module("lc_lifecycle")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def fixture_run(root):
    from lc_checkpoint import seal, snapshot_identity
    from protocol import file_sha256

    config = {
        "kind": "full-fit", "claim": "released-ID compatible-backbone hybrid",
        "recipe": {"epochs": 2, "world_size": 2},
        "geometry": {"updates_per_epoch": 2},
        "inputs": {"dataset": "fixture", "validation_examples": 6,
                   "format": {"validation_prompts": 2}},
    }
    write(root / "run-config.json", config)
    write(root / "runtime.json", {"pid": 123})
    write(root / "loader" / "run-config.json", {"microbatches_per_epoch": 2})
    calls = [{"global_step": 2, "epoch": 1.0, "eval_loss": 0.5},
             {"global_step": 4, "epoch": 2.0, "eval_loss": 0.7}]
    for step in (2, 4):
        path = root / "snapshots" / f"checkpoint-{step}"
        path.mkdir(parents=True)
        for name in ("adapter_model.bin", "optimizer.pt", "scheduler.pt"):
            (path / name).write_bytes(b"fixture-" + str(step).encode())
        write(path / "adapter_config.json", {})
        write(path / "run-config.json", config)
        write(path / "trainer_state.json", {
            "global_step": step, "epoch": step / 2, "best_metric": 0.5,
            "best_model_checkpoint": str(root / "checkpoint-2"),
        })
        write(path / "callback-state.json", {"validation_calls": calls[:step // 2]})
        for rank in range(2):
            (path / f"rng_state_{rank}.pth").write_bytes(b"rng")
            (path / f"rank-state-{rank}.pt").write_bytes(b"rank")
        seal(path, 2)
        write(root / "acks" / f"checkpoint-{step}.json", {
            **snapshot_identity(path), "status": "verified", "writer_host": "node-0",
        })
    write(root / "completed.json", {
        "status": "complete", "kind": "full-fit", "global_step": 4, "epoch": 2.0,
        "validation_calls": calls, "selected_checkpoint": "checkpoint-2",
        "selected_complete_sha256": file_sha256(root / "snapshots" / "checkpoint-2" / "complete.json"),
        "run_config_sha256": file_sha256(root / "run-config.json"),
    })
    return config


def test_final_bundle_preserves_root_completion_and_both_checkpoint_acks(tmp_path):
    lib = lifecycle()
    run = tmp_path / "run"
    fixture_run(run)
    bundle = tmp_path / "final-bundle"
    lib.freeze_final_run(run, bundle)
    value = lib.verify_bundle(bundle, "final-run")
    assert (bundle / "run" / "completed.json").read_bytes() == (run / "completed.json").read_bytes()
    assert value["identity"]["selected"]["global_step"] == 2
    assert value["identity"]["final"]["global_step"] == 4
    assert value["identity"]["expected_test_users"] == 3
    assert "run/completed.json" in value["files"]
    assert "selected/ack.json" in value["files"] and "final/ack.json" in value["files"]


def test_checkpoint_ack_never_authorizes_final_test(tmp_path):
    lib = lifecycle()
    run = tmp_path / "run"
    fixture_run(run)
    bundle = tmp_path / "final-bundle"
    lib.freeze_final_run(run, bundle)
    checkpoint_ack = json.loads((run / "acks" / "checkpoint-2.json").read_text())
    with pytest.raises(ValueError, match="bundle|final-run"):
        lib.authorize_evaluation(run, bundle, checkpoint_ack, "author-position-sets")
    manifest = lib.verify_bundle(bundle, "final-run")
    from protocol import file_sha256
    bank_ack = {
        "kind": "lc-bundle-bank", "bundle_kind": "final-run", "status": "verified",
        "bundle_sha256": file_sha256(bundle / "bundle.json"), "identity": manifest["identity"],
        "writer_host": "node-0",
    }
    lib.authorize_evaluation(run, bundle, bank_ack, "author-position-sets")
    (run / "completed.json").write_text("{}")
    with pytest.raises(ValueError, match="completion|checksum"):
        lib.authorize_evaluation(run, bundle, bank_ack, "author-position-sets")


def test_final_bundle_rejects_missing_final_ack_and_partial_budget(tmp_path):
    lib = lifecycle()
    run = tmp_path / "run"
    fixture_run(run)
    (run / "acks" / "checkpoint-4.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        lib.freeze_final_run(run, tmp_path / "not-final")
    done = json.loads((run / "completed.json").read_text())
    done["global_step"] = 3
    write(run / "completed.json", done)
    with pytest.raises(ValueError, match="budget"):
        lib.freeze_final_run(run, tmp_path / "partial")


def test_queue_requires_banked_both_policies_then_real_gate():
    lib = lifecycle()
    facts = {
        "instruments_exited": True, "instruments_complete": True,
        "instruments_final_banked": False, "instruments_evaluations_banked": [],
        "yelp_cpu_pass": True, "yelp_gpu_pass": False, "gpus_free": True,
    }
    with pytest.raises(ValueError):
        lib.require_queue_gate("instruments-evaluate", facts)
    facts["instruments_final_banked"] = True
    lib.require_queue_gate("instruments-evaluate", facts)
    with pytest.raises(ValueError):
        lib.require_queue_gate("yelp-gpu-gate", facts)
    facts["instruments_evaluations_banked"] = ["author-position-sets"]
    with pytest.raises(ValueError):
        lib.require_queue_gate("yelp-gpu-gate", facts)
    facts["instruments_evaluations_banked"].append("full-prefix-trie-correction")
    lib.require_queue_gate("yelp-gpu-gate", facts)
    with pytest.raises(ValueError):
        lib.require_queue_gate("yelp-fit", facts)
    facts["yelp_gpu_pass"] = True
    lib.require_queue_gate("yelp-fit", facts)
    facts["gpus_free"] = False
    with pytest.raises(ValueError):
        lib.require_queue_gate("yelp-fit", facts)


def test_single_use_job_reservation_is_not_retried(tmp_path):
    lib = lifecycle()
    job = {"kind": "evaluation", "argv": ["python", "evaluate.py"], "cuda": "0,1,2,3"}
    path = tmp_path / "attempt"
    lib.reserve_job(path, job)
    with pytest.raises(FileExistsError):
        lib.reserve_job(path, copy.deepcopy(job))


def test_evaluation_bundle_contains_and_verifies_every_rank_record(tmp_path):
    lib = lifecycle()
    from protocol import file_sha256
    run = tmp_path / "run"
    fixture_run(run)
    final = tmp_path / "final"
    lib.freeze_final_run(run, final)
    manifest = lib.verify_bundle(final, "final-run")
    ack = tmp_path / "final-bank.json"
    write(ack, {"kind": "lc-bundle-bank", "bundle_kind": "final-run", "status": "verified",
                "bundle_sha256": file_sha256(final / "bundle.json"), "identity": manifest["identity"]})
    evaluation = tmp_path / "evaluation"
    policy = "author-position-sets"
    config = {"run_config_sha256": file_sha256(run / "run-config.json"), "decoder_policy": policy,
              "num_beams": 20, "drop_last": False, "seen_item_exclusion": False, "world_size": 2}
    write(evaluation / "run-config.json", config)
    for rank, users in ((0, ["0", "2"]), (1, ["1"])):
        records = evaluation / f"predictions-rank-{rank}.jsonl"
        records.write_text("".join(json.dumps({
            "user": user, "raw_codes": [str(i) for i in range(20)], "raw_scores": [-float(i) for i in range(20)],
        }) + "\n" for user in users))
        write(evaluation / f"completed-rank-{rank}.json", {
            "status": "complete", "users": len(users), "sha256": file_sha256(records),
        })
    write(evaluation / "completed.json", {
        "status": "complete", "decoder_policy": policy, "users": 3,
        "run_config_sha256": file_sha256(evaluation / "run-config.json"),
    })
    bundle = tmp_path / "eval-bundle"
    lib.freeze_evaluation(run, evaluation, bundle, policy, ack)
    stored = lib.verify_bundle(bundle, "evaluation")
    assert "evaluation/predictions-rank-0.jsonl" in stored["files"]
    assert "evaluation/predictions-rank-1.jsonl" in stored["files"]
    (evaluation / "predictions-rank-1.jsonl").write_text("")
    with pytest.raises(ValueError, match="checksum"):
        lib.freeze_evaluation(run, evaluation, tmp_path / "broken-evaluation", policy, ack)


@pytest.mark.skipif(os.name != "posix", reason="Bank uses Linux flock")
def test_final_bank_receipt_is_distinct_and_verifies_every_file(tmp_path):
    lib = lifecycle()
    run = tmp_path / "run"
    fixture_run(run)
    bundle = tmp_path / "final-bundle"
    lib.freeze_final_run(run, bundle)
    target = tmp_path / "bank" / "final-bundle"
    receipt = lib.bank_bundle(bundle, target, tmp_path / "bank.lock", socket.gethostname())
    assert receipt["bundle_kind"] == "final-run"
    lib.verify_bundle_ack(bundle, receipt, "final-run")
    (target / "run" / "completed.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        lib.bank_bundle(bundle, target, tmp_path / "bank.lock", socket.gethostname())
