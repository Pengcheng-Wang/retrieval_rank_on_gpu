"""Predeclared paired-user bootstrap for matched released-ID temperature arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from protocol import file_sha256, require_hash, write_json


METRICS = ("recall@5", "recall@10", "ndcg@5", "ndcg@10")


def paired_bootstrap(left: np.ndarray, right: np.ndarray, seed: int, replicates: int) -> dict:
    if (left.shape != right.shape or left.ndim != 2 or len(left) < 2 or left.shape[1] < 1
            or replicates < 2 or not np.isfinite(left).all() or not np.isfinite(right).all()
            or (left < 0).any() or (left > 1).any() or (right < 0).any() or (right > 1).any()):
        raise ValueError("Paired metrics must be aligned finite [0,1] matrices with at least two users")
    difference = left - right
    random = np.random.default_rng(seed)
    samples = np.empty((replicates, left.shape[1]), dtype=np.float64)
    for i in range(replicates):
        indices = random.integers(0, len(left), size=len(left))
        samples[i] = difference[indices].mean(axis=0)
    return {
        "left_mean": left.mean(axis=0).tolist(),
        "right_mean": right.mean(axis=0).tolist(),
        "difference": difference.mean(axis=0).tolist(),
        "ci_low": np.quantile(samples, 0.025, axis=0).tolist(),
        "ci_high": np.quantile(samples, 0.975, axis=0).tolist(),
        "difference_sd": difference.std(axis=0, ddof=1).tolist(),
        "wins": (difference > 0).sum(axis=0).tolist(),
        "ties": (difference == 0).sum(axis=0).tolist(),
        "losses": (difference < 0).sum(axis=0).tolist(),
    }


def load_arm(directory: Path):
    completed = json.loads((directory / "completed.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "evaluation-config.json").read_text(encoding="utf-8"))
    if completed["status"] != "complete" or config["beams"] != 20:
        raise ValueError("Only completed fixed-budget evaluation arms can be compared")
    require_hash(directory / "predictions.jsonl", completed["predictions_sha256"])
    rows = {}
    with (directory / "predictions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["user"] in rows:
                raise ValueError("Duplicate evaluation user")
            rows[row["user"]] = row
    if len(rows) != completed["users"] or len(rows) != config["expected_users"]:
        raise ValueError("Evaluation coverage is incomplete")
    return completed, config, rows


def compare_arms(left_path: Path, right_path: Path, seed: int, replicates: int) -> dict:
    left_done, left_config, left_rows = load_arm(left_path)
    right_done, right_config, right_rows = load_arm(right_path)
    if (set(left_rows) != set(right_rows) or left_config["split"] != right_config["split"]
            or left_config["dataset"] != right_config["dataset"]):
        raise ValueError("Arms do not have the same paired users, split and dataset")
    for field in ("dataset", "seed", "inputs", "author_source", "options", "world_size",
                  "initialization", "tokenizer_vocab_sha256", "model_config",
                  "harness_sha256", "trainer_source_sha256", "versions",
                  "experiment_lineage"):
        if left_config["source_run"][field] != right_config["source_run"][field]:
            raise ValueError(f"Unmatched training control: {field}")
    users = sorted(left_rows)
    for user in users:
        if any(left_rows[user][field] != right_rows[user][field] for field in ("target_item", "target_code")):
            raise ValueError("Paired targets differ")
    result = {
        "dataset": left_config["dataset"], "split": left_config["split"],
        "training_seed": left_config["source_run"]["seed"],
        "left_temperature": left_config["source_run"]["temperature"],
        "right_temperature": right_config["source_run"]["temperature"],
        "difference_direction": "left_minus_right",
        "paired_users": len(users), "bootstrap_seed": seed, "bootstrap_replicates": replicates,
        "method": "paired-user percentile bootstrap; marginal unadjusted 95% intervals",
        "user_order": "lexicographic string UID",
        "assumptions": [
            "One paired target per user; no missing, dropped or duplicate users.",
            "User resampling conditional on these two trained models; no normality assumption.",
            "Does not include training-seed uncertainty or establish population/production equivalence.",
            "Shared-item and other cross-user dependence is not separately modeled.",
            "No p-value or family-wise significance claim; all prespecified metrics are reported."
        ],
        "left_predictions_sha256": left_done["predictions_sha256"],
        "right_predictions_sha256": right_done["predictions_sha256"],
        "metrics": {},
    }
    for kind in ("code", "item"):
        left = np.asarray([[left_rows[u]["metrics"][kind][m] for m in METRICS] for u in users])
        right = np.asarray([[right_rows[u]["metrics"][kind][m] for m in METRICS] for u in users])
        for position, metric in enumerate(METRICS):
            if not np.isclose(left[:, position].mean(), left_done["metrics"][kind][metric], rtol=0, atol=1e-12):
                raise ValueError("Left aggregate differs from per-user data")
            if not np.isclose(right[:, position].mean(), right_done["metrics"][kind][metric], rtol=0, atol=1e-12):
                raise ValueError("Right aggregate differs from per-user data")
        statistics = paired_bootstrap(left, right, seed, replicates)
        result["metrics"][kind] = {
            metric: {name: values[i] for name, values in statistics.items()}
            for i, metric in enumerate(METRICS)
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Comparison artifacts are immutable; use a new output")
    result = compare_arms(args.left, args.right, args.seed, args.replicates)
    result["analysis_sha256"] = file_sha256(Path(__file__))
    result["numpy_version"] = np.__version__
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
