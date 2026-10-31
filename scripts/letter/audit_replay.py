"""Independently recompute released-ID replay metrics before publishing aggregates."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from prepare import read_release
from protocol import file_sha256, require_hash, write_json


def audit_prediction(row: dict, target: int, item_codes: dict, groups: dict) -> dict:
    codes, scores = row["codes"], row["scores"]
    if (row["target_item"] != target or row["target_code"] != item_codes[target]
            or row["target_collision_size"] != len(groups[item_codes[target]])
            or len(codes) != 20 or len(set(codes)) != 20 or len(scores) != 20
            or any(code not in groups for code in codes)
            or not all(math.isfinite(score) for score in scores)
            or scores != sorted(scores, reverse=True)):
        raise ValueError("Prediction target, fixed beams, validity, collision group or scores failed audit")
    expanded = list(dict.fromkeys(item for code in codes for item in groups[code]))[:10]
    if expanded != row["item_predictions"]:
        raise ValueError("Original-item expansion does not obey the target-independent tie rule")
    result = {}
    for kind, predictions, truth in (("code", codes, item_codes[target]), ("item", expanded, target)):
        values = {}
        for k in (1, 5, 10):
            position = next((i + 1 for i, value in enumerate(predictions[:k]) if value == truth), None)
            values[f"recall@{k}"] = float(position is not None)
            values[f"ndcg@{k}"] = 0.0 if position is None else 1.0 / math.log2(position + 1)
        if (set(values) != set(row["metrics"][kind])
                or any(not math.isclose(value, row["metrics"][kind][name], rel_tol=0, abs_tol=1e-14)
                       for name, value in values.items())):
            raise ValueError("Stored per-user metrics differ from independent rank calculation")
        result[kind] = values
    return result


def verify_durable(source: Path, destination: Path, names: list[str]) -> None:
    receipt = json.loads((destination / "bank-receipt.json").read_text(encoding="utf-8"))
    for name in names:
        digest = file_sha256(source / name)
        if receipt["files"][name]["sha256"] != digest:
            raise ValueError("Durable bank receipt is bound to another artifact")
        require_hash(destination / name, digest)


def audit(args) -> dict:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    interactions, _, indices = read_release(args.upstream, manifest, args.dataset)
    item_codes = {int(item): "".join(tokens) for item, tokens in indices.items()}
    groups = defaultdict(list)
    for item in sorted(item_codes):
        groups[item_codes[item]].append(item)
    done = json.loads((args.evaluation / "completed.json").read_text(encoding="utf-8"))
    config = json.loads((args.evaluation / "evaluation-config.json").read_text(encoding="utf-8"))
    trained = json.loads((args.checkpoint / "completed.json").read_text(encoding="utf-8"))
    run = json.loads((args.checkpoint / "run-config.json").read_text(encoding="utf-8"))
    if (done["status"] != "complete" or trained["status"] != "complete"
            or config["source_run"] != run or config["checkpoint_sha256"] != trained["model_sha256"]
            or config["dataset"] != args.dataset or run["dataset"] != args.dataset
            or config["split"] != "test" or config["beams"] != 20
            or config["seen_item_exclusion"] is not False
            or config["expected_users"] != len(interactions)
            or run["inputs"] != manifest["datasets"][args.dataset]["sha256"]):
        raise ValueError("Completed model/evaluation and frozen input protocol do not agree")
    require_hash(args.evaluation / "predictions.jsonl", done["predictions_sha256"])
    require_hash(args.checkpoint / "pytorch_model.bin", trained["model_sha256"])
    require_hash(args.checkpoint / "run-config.json", trained["run_config_sha256"])
    seen, collisions = set(), 0
    totals = {"code": defaultdict(float), "item": defaultdict(float)}
    with (args.evaluation / "predictions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            user = row["user"]
            if user not in interactions or user in seen:
                raise ValueError("Unknown or duplicated evaluation user")
            metrics = audit_prediction(row, interactions[user][-1], item_codes, groups)
            seen.add(user)
            collisions += int(row["target_collision_size"] > 1)
            for kind, values in metrics.items():
                for name, value in values.items():
                    totals[kind][name] += value
    if seen != set(interactions) or done["users"] != len(seen):
        raise ValueError("Missing or unexpected test users")
    metrics = {kind: {k: v / len(seen) for k, v in values.items()} for kind, values in totals.items()}
    for kind, values in metrics.items():
        if set(values) != set(done["metrics"][kind]) or any(
            not math.isclose(value, done["metrics"][kind][key], rel_tol=0, abs_tol=1e-12)
            for key, value in values.items()
        ):
            raise ValueError("Aggregate metrics differ from the independently audited predictions")
    name = Path(trained["best_checkpoint"]).name
    if not re.fullmatch(r"checkpoint-\d+", name):
        raise ValueError("Invalid selected-checkpoint identity")
    selected = json.loads((args.checkpoint / name / "trainer_state.json").read_text())
    state = json.loads((args.checkpoint / "trainer_state.json").read_text())
    validation = [row for row in state["log_history"] if "eval_loss" in row]
    if (not validation or min(row["eval_loss"] for row in validation) != trained["best_metric"]
            or selected["best_metric"] != trained["best_metric"]
            or Path(selected["best_model_checkpoint"]).name != name):
        raise ValueError("Reported selected model does not agree with minimum validation loss")
    if args.bank_lock is not None:
        import fcntl
        with args.bank_lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            verify_durable(args.checkpoint, args.durable_model, [
                "pytorch_model.bin", "run-config.json", "completed.json",
            ])
            verify_durable(args.evaluation, args.durable_evaluation, [
                "predictions.jsonl", "evaluation-config.json", "completed.json",
            ])
    paper = manifest["published_results_not_reproduced_measurements"]
    if run["temperature"] == 0.7:
        reference = dict(zip(paper["table1_metric_order"], paper["table1"][args.dataset]["LETTER-TIGER"]))
        reference.update(source="https://arxiv.org/html/2405.07314v3#S4.T1",
                         row=f"LETTER-TIGER / {args.dataset}")
    elif run["temperature"] == 1.0 and args.dataset in paper["table2"]:
        reference = dict(zip(paper["table2_metric_order"], paper["table2"][args.dataset]["3"]))
        reference.update(source="https://arxiv.org/html/2405.07314v3#S4.T2",
                         row=f"all regularizers, ordinary loss / {args.dataset}")
    else:
        reference = {"row": "No matching published ordinary-loss reference"}
    return {
        "status": "verified_single_cell", "study_status": "partial", "dataset": args.dataset,
        "method": "LETTER-TIGER with author-released semantic IDs",
        "scope": "Released-ID replay; not an end-to-end tokenizer reconstruction",
        "seed": run["seed"], "temperature": run["temperature"],
        "learning_rate": run["options"]["learning_rate"], "world_size": run["world_size"],
        "per_device_batch_size": run["options"]["per_device_train_batch_size"],
        "gradient_accumulation_steps": run["options"]["gradient_accumulation_steps"],
        "selected_epoch": selected["epoch"], "stopped_epoch": trained["epoch"],
        "global_step": trained["global_step"],
        "checkpoint_selection": "minimum validation loss; patience20",
        "test_users": len(seen), "target_users_in_semantic_id_collision_groups": collisions,
        "num_beams": 20, "all_users_once": True, "seen_item_exclusion": False,
        "code_metrics": metrics["code"],
        "original_item_diagnostic": {
            "tie_rule": "Preserve code rank; expand ascending original item IDs; deduplicate; truncate K.",
            **metrics["item"],
        },
        "paper_reference_not_measured_here": reference,
        "timing_scope": {
            "trainer_invocation_seconds": trained["wall_seconds_this_invocation"],
            "evaluation_loop_seconds": done["elapsed_seconds"],
            "excludes_environment_setup_and_input_audit": True,
            "resumed_invocation_is_not_total_training_time": True,
            "hardware": "NVIDIA A100 80GB", "training_gpus": run["world_size"], "evaluation_gpus": 1,
        },
        "provenance": {
            "author_commit": "8d0154e28de37dbb6e24871c508ad8ddb1921cda",
            "training_harness_sha256": run["harness_sha256"],
            "model_sha256": trained["model_sha256"], "predictions_sha256": done["predictions_sha256"],
            "input_sha256": run["inputs"], "audit_implementation_sha256": file_sha256(Path(__file__)),
            "independent_metric_audit": "Every original target, code/item rank, beam, score and user audited.",
            "durability": "Selected model and complete predictions independently checksum-verified"
                          if args.bank_lock is not None else "not checked by this invocation",
        },
        "limitations": [
            "Single training seed; no cross-seed robustness or statistical-equivalence claim.",
            "Released-ID upstream tokenizer fitting provenance is incomplete.",
            "Original-item diagnostics use a declared arbitrary tie rule, not a learned reranker.",
            "Missing baseline/fresh-tokenizer/LC-Rec measurements are not implied by this replay.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("upstream", "manifest", "checkpoint", "evaluation", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dataset", choices=("Instruments", "Beauty", "Yelp"), required=True)
    parser.add_argument("--durable-model", type=Path)
    parser.add_argument("--durable-evaluation", type=Path)
    parser.add_argument("--bank-lock", type=Path)
    args = parser.parse_args()
    if any((args.durable_model, args.durable_evaluation, args.bank_lock)) and not all(
        (args.durable_model, args.durable_evaluation, args.bank_lock)
    ):
        parser.error("Both durable artifacts and the shared bank lock must be supplied together")
    if args.output.exists():
        raise FileExistsError("Published audit outputs are immutable")
    result = audit(args)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
