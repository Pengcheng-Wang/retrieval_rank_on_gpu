"""Paired-user reporting for Figure 8 tokenizer panels against tau0.7 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare import METRICS, load_arm, paired_bootstrap
from figure8_tokenizer import ARM_ORDER, aggregate_panels, load_plan
from protocol import file_sha256, write_json


def verify_baseline_result(path: Path, expected_sha256: str) -> dict:
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError("Figure8 shared baseline result identity differs")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not str(result.get("status", "")).startswith("verified"):
        raise ValueError("Figure8 shared baseline result is not independently verified")
    return result


def compare_tokenizer_arm(
    arm_path: Path, baseline_path: Path, seed: int, replicates: int,
) -> dict:
    arm_done, arm_config, arm_rows = load_arm(arm_path)
    base_done, base_config, base_rows = load_arm(baseline_path)
    if (set(arm_rows) != set(base_rows)
            or arm_config["dataset"] != base_config["dataset"]
            or arm_config["split"] != base_config["split"]):
        raise ValueError("Figure8 arm and baseline do not have identical paired users")
    arm_run, base_run = arm_config["source_run"], base_config["source_run"]
    for field in (
        "dataset", "seed", "temperature", "author_source", "options",
        "world_size", "initialization", "model_config", "harness_sha256",
        "trainer_source_sha256", "versions",
    ):
        if arm_run[field] != base_run[field]:
            raise ValueError(f"Unmatched Figure8 downstream control: {field}")
    for kind in ("item", "inter"):
        if arm_run["inputs"][kind] != base_run["inputs"][kind]:
            raise ValueError(f"Unmatched Figure8 non-tokenizer input: {kind}")
    users = sorted(arm_rows)
    if any(
        arm_rows[user]["target_item"] != base_rows[user]["target_item"]
        for user in users
    ):
        raise ValueError("Figure8 paired original-item targets differ")
    result = {
        "difference_direction": "arm_minus_shared_tau0p7_baseline",
        "paired_users": len(users),
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "method": "paired-user percentile bootstrap; marginal unadjusted 95% intervals",
        "arm_predictions_sha256": arm_done["predictions_sha256"],
        "baseline_predictions_sha256": base_done["predictions_sha256"],
        "metrics": {},
    }
    for identity in ("code", "item"):
        left = np.asarray([
            [arm_rows[user]["metrics"][identity][metric] for metric in METRICS]
            for user in users
        ])
        right = np.asarray([
            [base_rows[user]["metrics"][identity][metric] for metric in METRICS]
            for user in users
        ])
        statistics = paired_bootstrap(left, right, seed, replicates)
        result["metrics"][identity] = {
            metric: {name: values[position] for name, values in statistics.items()}
            for position, metric in enumerate(METRICS)
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan.resolve())
    arms = {}
    for value in args.arm:
        arm, separator, path = value.partition("=")
        if not separator or arm not in ARM_ORDER or arm in arms:
            raise ValueError("Use each exact --arm ARM=EVALUATION_DIRECTORY once")
        arms[arm] = Path(path).resolve()
    if set(arms) != set(ARM_ORDER):
        raise ValueError("All 15 Figure8 tokenizer evaluations are required")
    if args.output.exists():
        raise FileExistsError("Figure8 tokenizer comparison is immutable")
    verify_baseline_result(
        args.baseline_result.resolve(),
        plan["baseline"]["result_sha256"],
    )
    baseline_done, baseline_config, _ = load_arm(args.baseline.resolve())
    if (baseline_config.get("dataset") != "Instruments"
            or baseline_config.get("split") != "test"):
        raise ValueError("Figure8 shared baseline evaluation identity differs")
    baseline = {
        "run_id": plan["baseline"]["run_id"],
        "temperature": 0.7,
        "predictions_sha256": baseline_done["predictions_sha256"],
        "metrics": baseline_done["metrics"],
        "evaluation_config_sha256":
            file_sha256(args.baseline / "evaluation-config.json"),
    }
    points = {}
    for arm in ARM_ORDER:
        done, _, _ = load_arm(arms[arm])
        points[arm] = {
            "metrics": done["metrics"],
            "predictions_sha256": done["predictions_sha256"],
            "evaluation_config_sha256":
                file_sha256(arms[arm] / "evaluation-config.json"),
            "contrast": compare_tokenizer_arm(
                arms[arm],
                args.baseline.resolve(),
                plan["bootstrap_seed"],
                plan["bootstrap_replicates"],
            ),
        }
    result = aggregate_panels(points, baseline)
    result.update({
        "plan_sha256": file_sha256(args.plan),
        "analysis_sha256": file_sha256(Path(__file__)),
        "baseline_result_sha256": plan["baseline"]["result_sha256"],
        "intervals": "paired-user percentile bootstrap; marginal and unadjusted",
    })
    write_json(args.output, result)


if __name__ == "__main__":
    main()
