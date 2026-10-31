"""Verify and reuse a published frozen-teacher reference without new inference."""

from __future__ import annotations

import ast
import json
import math
import platform
import shutil
from pathlib import Path

from bank import source_manifest
from prepare import read_release
from protocol import file_sha256, require_hash


SCRIPT_DIR = Path(__file__).resolve().parent
DEPENDENCIES = {"mechanisms.py", "cf.py", "prepare.py", "protocol.py"}
KIND = "frozen_32d_teacher_reference_only"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def runtime_signature():
    import numpy as np
    import torch

    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError("Frozen reference reuse requires deterministic CPU scoring")
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "device": "cpu",
        "precision": str(torch.get_default_dtype()).split(".")[-1],
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
    }


def scoring_ast(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    # Only the caller and CLI may change to introduce reference reuse.
    tree.body = [node for node in tree.body if not (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"analyze", "main"}
    )]
    return ast.dump(tree, include_attributes=False)


def verified_reference(args):
    bundle = Path(args.cf_reference)
    output = Path(args.output).resolve()
    if (bundle.is_symlink() or output == bundle.resolve()
            or output in bundle.resolve().parents or bundle.resolve() in output.parents):
        raise ValueError("Analysis output must not overlap the frozen reference bundle")
    published = read(args.cf_reference_manifest)
    if published.get("kind") != "frozen_32d_cf_teacher_reference" or published.get("status") != "complete":
        raise ValueError("Reference needs the published frozen-teacher manifest")
    wrapper = read(bundle / "completed.json")
    payload, source = bundle / "payload", bundle / "source"
    if (wrapper.get("status") != "complete" or wrapper.get("kind") != "frozen_teacher_reference_bundle"
            or payload.is_symlink() or source.is_symlink()
            or source_manifest(payload) != wrapper.get("payload_manifest")
            or source_manifest(source) != wrapper.get("source_manifest")):
        raise ValueError("Frozen reference bundle inventory or checksum differs")
    require_hash(payload / "completed.json", published["evaluation_completed_sha256"])
    require_hash(payload / "run-config.json", published["evaluation_config_sha256"])
    done, config = read(payload / "completed.json"), read(payload / "run-config.json")
    if (wrapper.get("reference_completed_sha256") != published["evaluation_completed_sha256"]
            or done.get("status") != "complete" or done.get("kind") != KIND
            or config.get("kind") != KIND or done.get("test_evaluated") is not True
            or any(value.get(key) is not False for value in (done, config)
                   for key in ("quantized_replacement", "tokenizer_analyzed"))
            or done.get("run_config_sha256") != published["evaluation_config_sha256"]):
        raise ValueError("Reference is not a completed original-teacher-only evaluation")
    actual = source_manifest(payload)["files"]
    if set(actual) != set(done["files_sha256"]) | {"completed.json"}:
        raise ValueError("Reference payload has an unexpected file inventory")
    for name, digest in done["files_sha256"].items():
        if actual[name]["sha256"] != digest:
            raise ValueError("Reference payload checksum differs from its published completion")
    if (config["manifest_sha256"] != published["manifest_sha256"]
            or config["evaluator_sha256"] != published["evaluator_sha256"]
            or config["runner_sha256"] != published["runner_sha256"]
            or set(config["evaluator_sha256"]) != DEPENDENCIES):
        raise ValueError("Reference source or input manifest identity changed")
    require_hash(args.manifest, config["manifest_sha256"])
    for name, digest in config["evaluator_sha256"].items():
        require_hash(source / name, digest)
        if name != "mechanisms.py":
            require_hash(SCRIPT_DIR / name, digest)
    require_hash(source / "runner.py", config["runner_sha256"])
    if scoring_ast(source / "mechanisms.py") != scoring_ast(SCRIPT_DIR / "mechanisms.py"):
        raise ValueError("Current scoring source or globals differ from the frozen reference")
    if (args.device != "cpu" or config["runtime"] != published["runtime"]
            or config["runtime"] != runtime_signature()
            or config["runtime"]["precision"] != "float32"
            or config["runtime"]["threads"] != 2 or config["runtime"]["interop_threads"] != 1
            or config.get("history_limit") != 50 or config.get("seen_item_exclusion") is not False
            or config.get("tie_rule") != "original_item_id_ascending"
            or config.get("ranking_helper") != "mechanisms.rank_cf"):
        raise ValueError("Reference runtime or CPU scoring semantics differ")
    teacher = Path(args.teacher)
    binding = config["teachers"][args.dataset]
    for name, key in (
        ("completed.json", "completion_sha256"), ("run-config.json", "config_sha256"),
        ("best.pt", "selected_teacher_sha256"), ("item-embeddings.pt", "item_embeddings_sha256"),
    ):
        require_hash(teacher / name, binding[key])
    teacher_done, teacher_config = read(teacher / "completed.json"), read(teacher / "run-config.json")
    manifest = read(args.manifest)
    interactions, items, _ = read_release(args.upstream, manifest, args.dataset)
    if (teacher_done.get("status") != "complete" or teacher_done.get("smoke") is not False
            or teacher_done.get("test_evaluated") is not False or binding.get("fit_test_evaluated") is not False
            or teacher_config.get("kind") != "training-only-BCE-teacher-reconstruction"
            or teacher_config.get("dataset") != args.dataset or teacher_config.get("epochs") != 201
            or teacher_config.get("seed") != 42 or teacher_config["model"]["hidden_units"] != 32
            or teacher_config["model"]["maxlen"] != 50
            or teacher_config["input_sha256"] != manifest["datasets"][args.dataset]["sha256"]["inter"]
            or teacher_done["epoch"] != binding["selected_epoch"]):
        raise ValueError("Reference names a different teacher or validation-selected generation")
    result = published["results"][args.dataset]
    directory = payload / args.dataset
    if (done["results"][args.dataset] != result
            or read(directory / "completed.json") != {"status": "complete", **result}
            or result["teacher_sha256"] != teacher_done["teacher_sha256"]
            or result["selected_epoch"] != teacher_done["epoch"]
            or result["items"] != len(items) or result["users"] != len(interactions)
            or result.get("device") != "cpu" or result.get("history_limit") != 50
            or result.get("seen_item_exclusion") is not False
            or result.get("tie_rule") != "original_item_id_ascending"
            or result.get("independently_recomputed") is not True):
        raise ValueError("Reference result, catalog or teacher binding differs")
    admission, access = read(payload / "admission.json"), read(directory / "test-access.json")
    if (admission.get("status") != "pass" or admission.get("test_scored") is not False
            or admission.get("validation_probe_users_per_dataset") != 100
            or admission.get("run_config_sha256") != published["evaluation_config_sha256"]
            or access.get("status") != "started" or access.get("fit_receipt_unchanged") is not True
            or access.get("teacher_sha256") != teacher_done["teacher_sha256"]
            or access.get("admission_sha256") != file_sha256(payload / "admission.json")):
        raise ValueError("Reference was not admitted before its separate test access")
    prediction_path = directory / "predictions.json"
    require_hash(prediction_path, result["predictions_sha256"])
    rows = read(prediction_path)
    users = sorted(interactions, key=int)
    if len(rows) != len(users) or [row["user"] for row in rows] != users:
        raise ValueError("Frozen reference lost or reordered evaluation users")
    totals = {f"{metric}@{k}": 0.0 for metric in ("recall", "ndcg") for k in (5, 10)}
    for user, row in zip(users, rows):
        predictions = row["top10"]
        target = row["target_item"]
        if (type(target) is not int or target != interactions[user][-1]
                or len(predictions) != 10 or len(set(predictions)) != 10
                or any(type(item) is not int or not 0 <= item < len(items) for item in predictions)):
            raise ValueError("Frozen reference target or original-item ranking differs")
        for k in (5, 10):
            if target in predictions[:k]:
                totals[f"recall@{k}"] += 1
                totals[f"ndcg@{k}"] += 1 / math.log2(predictions.index(target) + 2)
    metrics = {key: value / len(users) for key, value in totals.items()}
    if metrics != result["metrics"]:
        raise ValueError("Frozen reference metrics fail independent recomputation")
    provenance = {
        "published_manifest_sha256": file_sha256(args.cf_reference_manifest),
        "reference_completed_sha256": published["evaluation_completed_sha256"],
        "teacher_sha256": result["teacher_sha256"],
        "original_scoring_module_sha256": config["evaluator_sha256"]["mechanisms.py"],
        "current_analysis_sha256": file_sha256(SCRIPT_DIR / "mechanisms.py"),
        "consumer_sha256": file_sha256(Path(__file__)),
        "compatibility": "Identical module AST outside analyze/main; exact helper bytes and recorded runtime.",
        "batch_size": 128,
    }
    return result, prediction_path, provenance


def reuse_reference(args):
    result, source, provenance = verified_reference(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "cf-original-predictions.json"
    with source.open("rb") as original, target.open("xb") as copied:
        shutil.copyfileobj(original, copied)
    require_hash(target, result["predictions_sha256"])
    return {
        **{key: result[key] for key in (
            "users", "metrics", "seen_item_exclusion", "tie_rule", "history_limit",
            "device", "predictions_sha256",
        )},
        "reused_frozen_reference": True, "reference_provenance": provenance,
    }
