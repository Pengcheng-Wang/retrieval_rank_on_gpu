"""CPU contracts; real native int8 admission is a separate, mandatory gate."""

import copy
import importlib
import json
import math
import os
import socket
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))


def module(name):
    assert importlib.util.find_spec(name) is not None, f"Missing LC execution module: {name}"
    return importlib.import_module(name)


def test_instruments_preserves_author_geometry():
    runner = module("lc_run")
    manifest = json.loads((ROOT / "configs" / "letter-reproduction-v1.json").read_text())
    recipe = runner.recipe_for(manifest, "Instruments")
    assert (recipe["micro_batch"], recipe["accumulation"], recipe["world_size"]) == (8, 2, 4)
    assert recipe["effective_batch"] == 64
    assert recipe["epochs"] == 4
    assert recipe["temperature"] == 0.8
    assert runner.epoch_geometry(131837, 8, 4, 2) == {
        "examples": 131837, "microbatches_per_epoch": 4120,
        "updates_per_epoch": 2060, "effective_batch": 64,
        "distributed_padding_examples": 3,
    }


def test_unsafe_yelp_geometry_and_beauty_are_not_admitted():
    runner = module("lc_run")
    manifest = json.loads((ROOT / "configs" / "letter-reproduction-v1.json").read_text())
    with pytest.raises(ValueError, match="integral"):
        runner.epoch_geometry(225061, 32, 4, 2)
    with pytest.raises(ValueError, match="Beauty"):
        runner.recipe_for(manifest, "Beauty")
    with pytest.raises(ValueError, match="Yelp"):
        runner.recipe_for(manifest, "Yelp")


def test_decoder_uses_generation_width_not_response_substrings():
    evaluator = module("lc_evaluate")
    codes = [[10, 20, 30, 40], [10, 21, 31, 41], [11, 20, 30, 40]]
    native = evaluator.make_constraint(codes, 2, 7, "author-position-sets")
    trie = evaluator.make_constraint(codes, 2, 7, "full-prefix-trie-correction")
    prompt = [91, 92, 93, 91, 92, 93, 94]
    assert native(0, prompt) == [10, 11]
    assert native(0, prompt + [11]) == [20, 21]
    assert trie(0, prompt + [11]) == [20]
    assert trie(0, prompt + [10, 20, 30, 40]) == [2]
    assert trie(0, prompt + [10, 20, 30, 40, 2]) == [2]
    with pytest.raises(ValueError, match="prefix"):
        trie(0, prompt + [11, 21])
    with pytest.raises(ValueError, match="width"):
        native(0, prompt[:-1])


def test_native_code_metrics_keep_duplicates_item_diagnostic_deduplicates():
    evaluator = module("lc_evaluate")
    indices = {"0": ["a", "b", "c", "d"], "1": ["a", "b", "c", "d"],
               "2": ["e", "f", "g", "h"]}
    result = evaluator.score_beams(
        ["invalid", "abcd", "abcd", "efgh"], [-0.1, -0.2, -0.3, -0.4], 1, indices,
    )
    assert result["raw_codes"] == ["invalid", "abcd", "abcd", "efgh"]
    assert result["invalid_codes"] == 1
    assert result["duplicate_codes"] == 1
    assert result["code"]["recall@5"] == 1.0
    assert result["code"]["ndcg@5"] == pytest.approx(1 / math.log2(3) + 1 / math.log2(4))
    assert result["item_predictions"] == [0, 1, 2]
    assert result["item"]["ndcg@5"] == pytest.approx(1 / math.log2(3))
    with pytest.raises(ValueError, match="finite"):
        evaluator.score_beams(["abcd"], [float("nan")], 1, indices)


def make_checkpoint(path, world=2):
    path.mkdir()
    contents = {
        "adapter_model.bin": b"task-produced adapter",
        "adapter_config.json": b"{}",
        "optimizer.pt": b"task-produced optimizer",
        "scheduler.pt": b"task-produced scheduler",
        "callback-state.json": b'{"validation_calls": []}',
        "run-config.json": b'{"claim": "released-ID compatible-backbone hybrid"}',
        "trainer_state.json": b'{"global_step": 2, "epoch": 1.0}',
    }
    for rank in range(world):
        contents[f"rng_state_{rank}.pth"] = b"rng"
        contents[f"rank-state-{rank}.pt"] = b"sampler and ordered inputs"
    for name, value in contents.items():
        (path / name).write_bytes(value)


def test_checkpoint_requires_every_rank_and_detects_mutation(tmp_path):
    checkpoint = module("lc_checkpoint")
    path = tmp_path / "checkpoint-2"
    make_checkpoint(path)
    (path / "rng_state_1.pth").unlink()
    with pytest.raises(ValueError, match="rank|RNG"):
        checkpoint.seal(path, 2)
    (path / "rng_state_1.pth").write_bytes(b"rng")
    receipt = checkpoint.seal(path, 2)
    assert receipt["global_step"] == 2
    assert checkpoint.verify(path)["world_size"] == 2
    (path / "optimizer.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        checkpoint.verify(path)


def test_snapshot_is_independent_immutable_and_ack_is_hash_bound(tmp_path):
    checkpoint = module("lc_checkpoint")
    path = tmp_path / "checkpoint-2"
    make_checkpoint(path)
    checkpoint.seal(path, 2)
    snapshot = tmp_path / "snapshots" / "checkpoint-2"
    checkpoint.snapshot(path, snapshot)
    (path / "optimizer.pt").write_bytes(b"changed source")
    assert checkpoint.verify(snapshot)["global_step"] == 2
    with pytest.raises(FileExistsError):
        checkpoint.snapshot(path, snapshot)
    identity = checkpoint.snapshot_identity(snapshot)
    ack = {**identity, "status": "verified", "writer_host": "node-0"}
    checkpoint.verify_ack(snapshot, ack)
    ack["adapter_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="acknowledgment"):
        checkpoint.verify_ack(snapshot, ack)


def test_audit_contract_refuses_probe_incomplete_or_modified_receipts(tmp_path):
    runner = module("lc_run")
    from protocol import file_sha256
    root = tmp_path / "audit"
    root.mkdir()
    config = {"dataset": "Instruments", "probe_base_records": None}
    audit = {"status": "pass", "corpus_admission": "PASS", "probe": False,
             "dataset": "Instruments", "coverage_status": "pass",
             "all_fingerprints_reverified_after_audit": True}
    (root / "run-config.json").write_text(json.dumps(config))
    audit["run_config_sha256"] = file_sha256(root / "run-config.json")
    (root / "audit.json").write_text(json.dumps(audit))
    marker = {"status": "pass", "corpus_admission": "PASS", "files_sha256": {
        name: file_sha256(root / name) for name in ("audit.json", "run-config.json")
    }}
    (root / "completed.json").write_text(json.dumps(marker))
    assert runner.read_admitted_audit(root, "Instruments")[1] == config
    changed = copy.deepcopy(marker)
    changed["corpus_admission"] = "NOT_ADMITTED_PROBE"
    (root / "completed.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="full.*PASS"):
        runner.read_admitted_audit(root, "Instruments")
    (root / "completed.json").write_text(json.dumps(marker))
    (root / "audit.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        runner.read_admitted_audit(root, "Instruments")


def test_deep_continuation_comparison_is_bitwise_not_close():
    checkpoint = module("lc_checkpoint")
    torch = pytest.importorskip("torch")
    import numpy as np
    a = {"state": [torch.tensor([1.0]), np.array([1, 2]), (3, "test")]}
    checkpoint.assert_identical(a, copy.deepcopy(a))
    b = copy.deepcopy(a)
    b["state"][0][0] += 1e-7
    with pytest.raises(ValueError, match="state"):
        checkpoint.assert_identical(a, b)


def test_bank_discovery_ignores_snapshot_construction_directories():
    checkpoint = module("lc_checkpoint")
    assert hasattr(checkpoint, "published_snapshot_names"), "Broker needs atomic publication filtering"
    names = checkpoint.published_snapshot_names("\n".join([
        "/owned/snapshots/checkpoint-50.building-912/complete.json",
        "/owned/snapshots/checkpoint-1/complete.json",
        "/owned/snapshots/checkpoint-100/complete.json",
        "/owned/snapshots/checkpoint-50/complete.json",
    ]))
    assert names == ["checkpoint-1", "checkpoint-50", "checkpoint-100"]


@pytest.mark.skipif(os.name != "posix", reason="Production bank uses Linux flock")
def test_lc_bank_roundtrip_is_immutable_and_writer_guarded(tmp_path):
    checkpoint = module("lc_checkpoint")
    source = tmp_path / "checkpoint-2"
    make_checkpoint(source)
    checkpoint.seal(source, 2)
    destination = tmp_path / "bank" / "checkpoint-2"
    lock = tmp_path / "bank.lock"
    with pytest.raises(RuntimeError, match="designated"):
        checkpoint.bank_one(source, destination, lock, "not-this-host")
    receipt = checkpoint.bank_one(source, destination, lock, socket.gethostname())
    assert receipt["bank_helper_sha256"]
    checkpoint.verify_ack(source, receipt)
    checkpoint.verify_ack(destination, receipt)
    assert checkpoint.bank_one(source, destination, lock, socket.gethostname()) == receipt
    (destination / "optimizer.pt").write_bytes(b"changed durable file")
    with pytest.raises(ValueError, match="checksum"):
        checkpoint.bank_one(source, destination, lock, socket.gethostname())
