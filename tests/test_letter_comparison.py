import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def compare():
    assert (SCRIPTS / "compare.py").exists(), "Paired result comparison is missing"
    import compare
    return compare


def test_paired_bootstrap_preserves_constant_within_user_difference():
    left = np.array([[0.0], [0.25], [0.5], [0.75]])
    result = compare().paired_bootstrap(left, left + 0.125, 20261027, 200)
    assert result["difference"] == [-0.125]
    assert result["ci_low"] == result["ci_high"] == [-0.125]


def test_identical_models_have_zero_observed_difference():
    values = np.array([[0, 0.0], [1, 0.5], [0, 0.0], [1, 1.0]])
    result = compare().paired_bootstrap(values, values, 42, 200)
    assert result["difference"] == [0.0, 0.0]
    assert result["ci_low"] == result["ci_high"] == [0.0, 0.0]


def test_paired_bootstrap_is_deterministic_and_rejects_bad_inputs():
    left = np.array([[0.0], [1.0], [0.0]])
    right = np.array([[1.0], [0.0], [0.0]])
    assert compare().paired_bootstrap(left, right, 7, 100) == compare().paired_bootstrap(left, right, 7, 100)
    with pytest.raises(ValueError):
        compare().paired_bootstrap(left, right[:2], 7, 100)
    with pytest.raises(ValueError):
        compare().paired_bootstrap(left * np.nan, right, 7, 100)


def write_arm(path: Path, lineage: dict) -> None:
    source_run = {
        "dataset": "Instruments",
        "seed": 42,
        "temperature": 1.0,
        "inputs": {"train": "a" * 64},
        "author_source": {"data.py": "b" * 64},
        "options": {"num_train_epochs": 200},
        "world_size": 2,
        "initialization": "author random T5 configuration",
        "tokenizer_vocab_sha256": "c" * 64,
        "model_config": {"d_model": 128},
        "harness_sha256": {"run.py": "d" * 64},
        "trainer_source_sha256": "e" * 64,
        "versions": {"torch": "1.13.1"},
        "experiment_lineage": lineage,
    }
    rows = [
        {
            "user": str(index),
            "target_item": index,
            "target_code": f"<a_{index}>",
            "metrics": {
                "code": {
                    "recall@1": 1.0, "recall@5": 1.0, "recall@10": 1.0,
                    "ndcg@5": 1.0, "ndcg@10": 1.0,
                },
                "item": {
                    "recall@1": 1.0, "recall@5": 1.0, "recall@10": 1.0,
                    "ndcg@5": 1.0, "ndcg@10": 1.0,
                },
            },
        }
        for index in range(2)
    ]
    path.mkdir()
    predictions = path / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    digest = __import__("hashlib").sha256(predictions.read_bytes()).hexdigest()
    (path / "evaluation-config.json").write_text(json.dumps({
        "dataset": "Instruments", "split": "test", "beams": 20,
        "expected_users": len(rows), "source_run": source_run,
    }))
    metrics = {
        kind: {
            "recall@1": 1.0, "recall@5": 1.0, "recall@10": 1.0,
            "ndcg@5": 1.0, "ndcg@10": 1.0,
        }
        for kind in ("code", "item")
    }
    (path / "completed.json").write_text(json.dumps({
        "status": "complete", "users": len(rows),
        "predictions_sha256": digest, "metrics": metrics,
    }))


def test_comparison_rejects_mismatched_experiment_lineage(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    write_arm(left, {"index_sha256": "1" * 64})
    write_arm(right, {"index_sha256": "2" * 64})

    with pytest.raises(ValueError, match="experiment_lineage"):
        compare().compare_arms(left, right, 7, 20)
