import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def module():
    assert (SCRIPTS / "protocol.py").exists(), "LETTER protocol implementation is missing"
    import protocol
    return protocol


def test_leave_two_out_preserves_event_identity():
    rows = module().sequence_records({"7": [0, 1, 2, 3, 4]}, "train")
    assert [(r["history"], r["target"]) for r in rows] == [([0], 1), ([0, 1], 2)]
    assert module().sequence_records({"7": [0, 1, 2, 3, 4]}, "valid") == [
        {"user": "7", "position": 3, "history": [0, 1, 2], "target": 3}
    ]
    assert module().sequence_records({"7": [0, 1, 2, 3, 4]}, "test") == [
        {"user": "7", "position": 4, "history": [0, 1, 2, 3], "target": 4}
    ]


def test_future_target_does_not_change_fitting_examples():
    protocol = module()
    before = protocol.sequence_records({"1": [0, 1, 2, 3, 4]}, "train")
    after = protocol.sequence_records({"1": [0, 1, 2, 3, 9]}, "train")
    assert before == after
    repeated = protocol.sequence_records({"1": [0, 1, 0, 2, 0]}, "test")
    assert repeated[0]["target"] in repeated[0]["history"]


def test_recent_history_window_and_invalid_inputs():
    protocol = module()
    row = protocol.sequence_records({"u": list(range(30))}, "test")[0]
    assert row["history"] == list(range(9, 29))
    with pytest.raises(ValueError):
        protocol.sequence_records({"u": [0, 1]}, "train")
    with pytest.raises(ValueError):
        protocol.sequence_records({"u": [0, 1, 2, 3, 4]}, "unknown")


def test_collision_metric_and_item_diagnostic():
    protocol = module()
    indices = {"0": ["a", "b"], "1": ["a", "b"], "2": ["c", "d"]}
    assert protocol.collision_summary(indices) == {
        "items": 3, "unique_sids": 2, "excess_collision_items": 1,
        "collision_groups": 1, "items_in_collision_groups": 2, "largest_group": 2,
    }
    assert protocol.expand_item_predictions(["ab", "bad", "ab", "cd"], indices, 3) == [0, 1, 2]
    result = protocol.score_prediction(["ab", "cd"], 1, indices)
    assert result["code"]["recall@1"] == 1
    assert result["item"]["recall@1"] == 0
    assert result["target_collision_size"] == 2


def test_single_target_metrics_and_duplicate_predictions():
    protocol = module()
    first = protocol.rank_metrics(["x", "target"], "target")
    absent = protocol.rank_metrics(["x"], "target")
    assert (first["recall@5"] + absent["recall@5"]) / 2 == 0.5
    assert first["ndcg@10"] / 2 == pytest.approx(0.5 / math.log2(3))
    duplicates = protocol.rank_metrics(["target", "target"], "target")
    assert duplicates["ndcg@10"] == 1


def test_prefix_trie_observes_full_prefix():
    trie = module().PrefixTrie([[0, 10, 20, 1], [0, 11, 21, 1]])
    assert trie.allowed([0]) == [10, 11]
    assert trie.allowed([0, 10]) == [20]
    assert trie.allowed([0, 10, 21]) == []


def test_complete_user_shards_do_not_drop_or_duplicate():
    protocol = module()
    shards = [protocol.shard_positions(5, r, 4) for r in range(4)]
    assert sorted(i for s in shards for i in s) == list(range(5))
    with pytest.raises(ValueError):
        protocol.shard_positions(5, 4, 4)
    protocol.verify_coverage(["u0", "u1"], ["u1", "u0"])
    with pytest.raises(ValueError, match="coverage"):
        protocol.verify_coverage(["u0", "u1"], ["u0", "u0"])


def test_resume_requires_all_hashed_files(tmp_path):
    protocol = module()
    ckpt = tmp_path / "checkpoint-5"
    ckpt.mkdir()
    for name in ("pytorch_model.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (ckpt / name).write_bytes(name.encode())
    (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": 5}))
    protocol.seal_checkpoint(ckpt)
    assert protocol.verify_checkpoint(ckpt) == 5
    (ckpt / "optimizer.pt").write_bytes(b"interrupted")
    with pytest.raises(ValueError, match="checksum"):
        protocol.verify_checkpoint(ckpt)


def test_weights_only_is_not_a_complete_resume(tmp_path):
    protocol = module()
    (tmp_path / "pytorch_model.bin").write_bytes(b"weights")
    with pytest.raises(ValueError, match="incomplete"):
        protocol.seal_checkpoint(tmp_path)


def test_effective_batch_and_source_integrity_fail_closed(tmp_path):
    protocol = module()
    assert protocol.accumulation_steps(1024, 256, 2) == 2
    with pytest.raises(ValueError):
        protocol.accumulation_steps(64, 16, 8)
    source = tmp_path / "source.py"
    source.write_text("unchanged")
    digest = protocol.file_sha256(source)
    protocol.require_hash(source, digest)
    source.write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        protocol.require_hash(source, digest)


def test_author_data_matches_canonical_splits(tmp_path):
    assert (SCRIPTS / "prepare.py").exists(), "Author-code preparation adapter is missing"
    from prepare import compare_author_rows
    records = module().sequence_records({"7": [0, 1, 2, 3, 4]}, "train")
    indices = {str(i): [f"<a_{i}>"] for i in range(5)}
    authored = [
        {"input_ids": "<a_0>", "labels": "<a_1>"},
        {"input_ids": "<a_0><a_1>", "labels": "<a_2>"},
    ]
    compare_author_rows(authored, records, indices, "tiger")
    authored[0]["labels"] = "<a_4>"
    with pytest.raises(ValueError, match="split"):
        compare_author_rows(authored, records, indices, "tiger")


def test_full_length_gate_rejects_lost_eos_and_empty_labels():
    assert (SCRIPTS / "prepare.py").exists(), "Full token-length gate is missing"
    from prepare import validate_token_rows
    assert validate_token_rows([[3, 4, 1]], [[8, 1]], 512, 1)["records"] == 1
    with pytest.raises(ValueError, match="EOS"):
        validate_token_rows([[3, 1]], [[8]], 512, 1)
    with pytest.raises(ValueError, match="length"):
        validate_token_rows([[3] * 513], [[8, 1]], 512, 1)
    with pytest.raises(ValueError, match="empty"):
        validate_token_rows([[3, 1]], [[]], 512, 1)


def test_source_loader_refuses_changed_backend(tmp_path):
    assert (SCRIPTS / "author.py").exists(), "Pinned source loader is missing"
    from author import verify_backend_source
    (tmp_path / "LETTER-TIGER").mkdir()
    (tmp_path / "LETTER-TIGER" / "data.py").write_text("different")
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_backend_source(tmp_path, "tiger")


def test_training_recipe_preserves_release_settings():
    assert (SCRIPTS / "run.py").exists(), "Author training orchestrator is missing"
    from run import training_options
    options = training_options(seed=42, batch_size=256, effective_batch=1024, world_size=2,
                               learning_rate=5e-4, epochs=200)
    assert options["gradient_accumulation_steps"] == 2
    assert options["evaluation_strategy"] == "epoch"
    assert options["metric_for_best_model"] == "eval_loss"
    assert options["greater_is_better"] is False
    assert options["warmup_ratio"] == 0.01
    assert options["lr_scheduler_type"] == "cosine"
    assert options["weight_decay"] == 0.01
    assert not options["fp16"] and not options["bf16"]
    assert options["report_to"] == []


def test_evaluation_never_reduces_the_beam_budget():
    assert (SCRIPTS / "evaluate.py").exists(), "Fixed-budget evaluator is missing"
    from evaluate import generate_fixed

    class OutOfMemory:
        calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            assert kwargs["num_beams"] == kwargs["num_return_sequences"] == 20
            raise RuntimeError("CUDA out of memory")

    model = OutOfMemory()
    with pytest.raises(RuntimeError, match="out of memory"):
        generate_fixed(model, {}, lambda batch, sent: [1], 20)
    assert model.calls == 1


def test_durable_copy_is_verified_immutable_and_idempotent(tmp_path):
    assert (SCRIPTS / "bank.py").exists(), "Durable artifact banking is missing"
    from bank import copy_artifact
    source = tmp_path / "source"
    source.mkdir()
    (source / "tensor.bin").write_bytes(b"complete tensor")
    destination = tmp_path / "durable" / "attempt-1"
    receipt = copy_artifact(source, destination)
    assert receipt == copy_artifact(source, destination)
    (source / "tensor.bin").write_bytes(b"changed tensor")
    with pytest.raises(ValueError, match="immutable"):
        copy_artifact(source, destination)


def test_durable_copy_refuses_unsealed_checkpoint(tmp_path):
    assert (SCRIPTS / "bank.py").exists(), "Durable artifact banking is missing"
    from bank import require_bankable
    checkpoint = tmp_path / "checkpoint-5"
    checkpoint.mkdir()
    (checkpoint / "pytorch_model.bin").write_bytes(b"partial")
    with pytest.raises(ValueError, match="incomplete"):
        require_bankable(checkpoint)


def test_nonintegral_accumulation_epochs_are_rejected():
    from run import require_integral_epoch
    require_integral_epoch(258, 2)
    require_integral_epoch(129, 1)
    with pytest.raises(ValueError, match="accumulation"):
        require_integral_epoch(257, 2)
    with pytest.raises(ValueError, match="accumulation"):
        require_integral_epoch(3, 2)


def test_legacy_resume_patch_fails_closed_on_unknown_source():
    assert (SCRIPTS / "runtime_patch.py").exists(), "Verified legacy resume patch is missing"
    from runtime_patch import patched_trainer
    with pytest.raises(ValueError, match="checksum"):
        patched_trainer(b"unrecognized Trainer implementation")


def test_snapshot_is_independent_of_training_checkpoint_mutation(tmp_path):
    from bank import snapshot_checkpoint
    protocol = module()
    source = tmp_path / "checkpoint-7"
    source.mkdir()
    for name in ("pytorch_model.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (source / name).write_bytes(name.encode())
    (source / "trainer_state.json").write_text(json.dumps({"global_step": 7}))
    protocol.seal_checkpoint(source)
    snapshot = tmp_path / "snapshot"
    snapshot_checkpoint(source, snapshot)
    (source / "pytorch_model.bin").write_bytes(b"replaced by trainer")
    assert protocol.verify_checkpoint(snapshot) == 7


def test_evaluator_checks_run_binding_not_just_weight_bytes(tmp_path):
    from evaluate import verify_model_binding
    protocol = module()
    config = {"dataset": "Instruments", "inputs": {"index": "correct"}}
    protocol.write_json(tmp_path / "run-config.json", config)
    (tmp_path / "pytorch_model.bin").write_bytes(b"selected weights")
    receipt = {"status": "complete",
               "model_sha256": protocol.file_sha256(tmp_path / "pytorch_model.bin"),
               "run_config_sha256": protocol.file_sha256(tmp_path / "run-config.json")}
    verify_model_binding(tmp_path, receipt, config, "Instruments", {"index": "correct"})
    with pytest.raises(ValueError, match="identity"):
        verify_model_binding(tmp_path, receipt, config, "Instruments", {"index": "different"})
    protocol.write_json(tmp_path / "run-config.json", {**config, "seed": 99})
    with pytest.raises(ValueError, match="identity"):
        verify_model_binding(tmp_path, receipt, config, "Instruments", {"index": "correct"})


def test_incomplete_training_run_is_not_bankable_as_a_completed_artifact(tmp_path):
    from bank import require_bankable
    (tmp_path / "run-config.json").write_text("{}")
    with pytest.raises(ValueError, match="incomplete"):
        require_bankable(tmp_path)
