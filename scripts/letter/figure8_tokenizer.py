"""Frozen downstream contract for the 15 nonbaseline Figure 8 tokenizer arms."""

from __future__ import annotations

import json
import re
import copy
from pathlib import Path, PurePosixPath

import numpy as np

from protocol import file_sha256, write_json


ARM_ORDER = (
    "alpha-0p001", "alpha-0p01", "alpha-0p1",
    "beta-0p00001", "beta-0p001", "beta-0p01",
    "clusters-5", "clusters-15", "clusters-20",
    "levels-2", "levels-6", "levels-8",
    "codes-64", "codes-128", "codes-512",
)
PREFIXES = "abcdefgh"
PANEL_ORDER = ("alpha", "beta", "clusters", "levels", "codes")
FORBIDDEN_RESULT_FIELDS = {
    "winner", "best_arm", "best_temperature", "selected_by_test",
    "recommended_arm", "recommended_temperature",
}
SCIENTIFIC_SPEC_FIELDS = (
    "dataset", "codes", "epsilons", "dimension", "layers", "clusters",
    "alpha", "beta", "seed", "epochs", "batch_size", "eval_step",
)
LAUNCH_SPEC_FIELDS = set(SCIENTIFIC_SPEC_FIELDS) | {
    "cf", "checkpoint_every", "control", "embedding_receipt", "gpu", "host",
    "kind", "manifest", "method_spec", "output", "root", "semantic",
}
FIGURE8_TOKENIZER_HARNESS_SHA256 = (
    "21e3909b8ea84793779820db8442e83b12a8597e3caf9d294d9f18bcb95a4532"
)
KMEANS_COMPATIBILITY_ARMS = {"beta-0p001", "clusters-20"}


def baseline_tokenizer_spec() -> dict:
    return {
        "dataset": "Instruments",
        "codes": [256, 256, 256, 256],
        "epsilons": [0.0, 0.0, 0.0, 0.003],
        "dimension": 32,
        "layers": [2048, 1024, 512, 256, 128, 64],
        "clusters": 10,
        "alpha": 0.02,
        "beta": 0.0001,
        "seed": 42,
        "epochs": 20000,
        "batch_size": 1024,
        "eval_step": 2000,
    }


def arm_tokenizer_spec(plan: dict, arm: str) -> dict:
    if arm not in ARM_ORDER or arm not in plan["arms"]:
        raise ValueError("Unknown Figure8 tokenizer arm")
    spec = baseline_tokenizer_spec()
    factor, value = arm.split("-", 1)
    if factor == "alpha":
        spec["alpha"] = float(value.replace("p", "."))
    elif factor == "beta":
        spec["beta"] = float(value.replace("p", "."))
    elif factor == "clusters":
        spec["clusters"] = int(value)
    elif factor == "levels":
        levels = int(value)
        spec["codes"] = [256] * levels
        spec["epsilons"] = [0.0] * (levels - 1) + [0.003]
    elif factor == "codes":
        size = int(value)
        spec["codes"] = [size] * 4
    else:
        raise ValueError("Unknown Figure8 tokenizer factor")
    return spec


def validate_arm_launch_spec(spec: dict, expected: dict, arm: str) -> dict:
    if set(spec) != LAUNCH_SPEC_FIELDS:
        raise ValueError("Figure8 tokenizer operational specification schema differs")
    projection = {field: spec[field] for field in SCIENTIFIC_SPEC_FIELDS}
    if projection != expected:
        raise ValueError("Figure8 tokenizer scientific specification differs")
    root = PurePosixPath(spec["root"])
    control = root / "figure8-sweeps-v1" / "control"
    expected_paths = {
        "control": control,
        "manifest": control / "configs" / "letter-reproduction-v1.json",
        "method_spec":
            control / "configs" / "letter-openllama-v2-method-v1.json",
        "output": root / "figure8-sweeps-v1" / "runs" / arm,
        "semantic":
            root / "embeddings-openllama-v2-v1" / "Instruments"
            / "embeddings.npy",
        "embedding_receipt":
            root / "embeddings-openllama-v2-v1" / "Instruments"
            / "embedding-receipt.json",
    }
    cf_paths = {
        root / parent / "Instruments-cf-bce-s42" / "item-embeddings.pt"
        for parent in ("runs", "incoming")
    }
    if (not root.is_absolute()
            or root.parent != PurePosixPath("/scratch")
            or spec["kind"] != "figure8-one-factor-tokenizer-v1"
            or spec["checkpoint_every"] != 500
            or not re.fullmatch(r"node-\d+", spec["host"])
            or type(spec["gpu"]) is not int
            or spec["gpu"] < 0
            or not isinstance(spec["cf"], str)
            or PurePosixPath(spec["cf"]) not in cf_paths
            or PurePosixPath(spec["cf"]).as_posix() != spec["cf"]
            or any(
                not isinstance(spec[name], str)
                or PurePosixPath(spec[name]) != path
                or PurePosixPath(spec[name]).as_posix() != spec[name]
                for name, path in expected_paths.items()
            )):
        raise ValueError("Figure8 tokenizer operational specification differs")
    return projection


def validate_arm_run_config(
    config: dict, expected: dict, execution: dict, items: int,
) -> dict:
    if {field: config.get(field) for field in SCIENTIFIC_SPEC_FIELDS} != expected:
        raise ValueError("Figure8 tokenizer bundle scientific specification differs")
    levels = len(expected["codes"])
    runtime = {
        "clusters": expected["clusters"],
        "codebook_shapes": [
            [size, expected["dimension"]] for size in expected["codes"]
        ],
        "emitted_sid_length": levels,
        "epsilons": expected["epsilons"],
        "last_token_prefix": PREFIXES[levels - 1],
        "levels": levels,
    }
    if config.get("runtime_parameterization") != runtime:
        raise ValueError("Figure8 tokenizer runtime parameterization differs")
    dataset = execution["datasets"]["Instruments"]
    provenance = {
        "cf_provenance": "training_only_BCE_teacher_reconstruction",
        "cf_sha256": dataset["cf_sha256"],
        "cf_shape": [items, 32],
        "claim_type": execution["claim_type"],
        "embedding_receipt_sha256": dataset["embedding_receipt_sha256"],
        "learning_rate": execution["rq"]["options"]["learning_rate"],
        "method_spec_sha256": execution["method_spec_canonical_sha256"],
        "num_workers": execution["rq"]["options"]["num_workers"],
        "purpose": {
            "admissible_final_tokenizer": True,
            "kind": "tokenizer_reconstruction",
        },
        "selection": execution["rq"]["options"]["selection"],
        "semantic_sha256": dataset["semantic_sha256"],
        "semantic_shape": [items, execution["semantic_encoder"]["dimension"]],
        "source": execution["rq"]["source"],
        "synthetic_fixture": False,
        "track_id": execution["track_id"],
        "weight_decay": execution["rq"]["options"]["weight_decay"],
    }
    if (any(config.get(key) != value for key, value in provenance.items())
            or config.get("harness_sha256")
            != FIGURE8_TOKENIZER_HARNESS_SHA256):
        raise ValueError("Figure8 tokenizer bundle provenance differs")
    expected_keys = set(SCIENTIFIC_SPEC_FIELDS) | set(provenance) | {
        "harness_sha256", "runtime_parameterization",
    }
    if set(config) != expected_keys:
        raise ValueError("Figure8 tokenizer normalized bundle schema differs")
    return config


def tokenizer_cells() -> list[dict]:
    return [
        {
            "arm": arm,
            "dataset": "Instruments",
            "seed": 42,
            "temperature": 0.7,
            "learning_rate": 0.0005,
            "epochs": 200,
            "world_size": 2,
            "effective_batch_size": 1024,
            "beams": 20,
            "stage_id": f"Instruments-figure8-tokenizer-{arm}-ids-v1",
            "run_id": f"openllama-v2-figure8-tokenizer-Instruments-{arm}-s42-t0p7-v1",
        }
        for arm in ARM_ORDER
    ]


def contains_winner_field(value) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).lower() in FORBIDDEN_RESULT_FIELDS
            or contains_winner_field(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(contains_winner_field(child) for child in value)
    return False


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    versions = {
        "letter_figure8_tokenizer_downstream_v1":
            (1, "fresh-t5-figure8-tokenizer-v1"),
        "letter_figure8_tokenizer_downstream_v2":
            (2, "fresh-t5-figure8-tokenizer-v2"),
        "letter_figure8_tokenizer_downstream_v3":
            (3, "fresh-t5-figure8-tokenizer-v3"),
    }
    version = versions.get(plan.get("kind"))
    if (version is None or plan.get("schema_version") != version[0]
            or plan.get("work_basename") != version[1]):
        raise ValueError("Figure8 tokenizer plan version and work namespace differ")
    expected = {
        "dataset": "Instruments",
        "seed": 42,
        "temperature": 0.7,
        "learning_rate": 0.0005,
        "epochs": 200,
        "world_size": 2,
        "effective_batch_size": 1024,
        "beams": 20,
        "seen_item_exclusion": False,
        "max_new_tokens": 10,
        "arms": list(ARM_ORDER),
        "selection": "within-arm earliest minimum validation loss with patience20",
        "test_selection": "descriptive only; no winner or recipe selection",
        "metric_identities": ["code", "item"],
        "comparison_metrics": ["recall@5", "recall@10", "ndcg@5", "ndcg@10"],
        "bootstrap_seed": 20261027,
        "bootstrap_replicates": 2000,
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("Figure8 tokenizer downstream plan differs from the frozen scientific grid")
    if contains_winner_field(plan):
        raise ValueError("Figure8 tokenizer downstream plan may not contain a test-derived winner")
    baseline = plan.get("baseline", {})
    if (baseline.get("temperature") != 0.7
            or baseline.get("run_id") != "openllama-v2-fresh-t5-Instruments-full-s42-t07-v1"
            or not re.fullmatch(r"[a-f0-9]{64}", baseline.get("result_sha256", ""))):
        raise ValueError("Figure8 tokenizer baseline identity differs")
    return plan


def validate_identifier_bank(
    indices: dict, codes: np.ndarray, items: int, codebook_sizes: list[int],
) -> dict:
    levels = len(codebook_sizes)
    if not 1 <= levels <= len(PREFIXES):
        raise ValueError("Identifier levels must be between one and eight")
    if (set(indices) != set(map(str, range(items)))
            or any(not isinstance(indices[str(item)], list) for item in range(items))):
        raise ValueError("Identifier bank must cover the complete contiguous catalog")
    if not isinstance(codes, np.ndarray) or codes.shape != (items, levels):
        raise ValueError(f"Identifier code shape must be ({items}, {levels})")
    if codes.dtype.kind not in "iu":
        raise ValueError("Identifier codes must be integers")
    if any(type(size) is not int or size < 1 for size in codebook_sizes):
        raise ValueError("Codebook sizes must be positive integers")
    for level, size in enumerate(codebook_sizes):
        column = codes[:, level]
        if np.any(column < 0) or np.any(column >= size):
            raise ValueError(f"Identifier level {level} range exceeds codebook size {size}")
    for item, row in enumerate(codes.tolist()):
        tokens = indices[str(item)]
        if len(tokens) != levels:
            raise ValueError("Identifier token count differs from requested levels")
        for level, value in enumerate(row):
            expected = f"<{PREFIXES[level]}_{value}>"
            if tokens[level] != expected:
                raise ValueError(
                    f"Identifier prefix/value differs at item {item}, level {level}"
                )
    return {
        "items": items,
        "levels": levels,
        "codebook_sizes": list(codebook_sizes),
        "prefixes": list(PREFIXES[:levels]),
        "target_tokens": levels + 1,
    }


def validate_variable_audit(report: dict, bank: dict, expected_records) -> dict:
    if report.get("status") != "pass":
        raise ValueError("Variable identifier audit did not pass")
    target_tokens = bank["levels"] + 1
    for split in ("train", "valid", "test"):
        part = report.get("splits", {}).get(split, {})
        count = expected_records[split] if isinstance(expected_records, dict) else expected_records
        if (part.get("records") != count
                or part.get("strict_clean") != count
                or part.get("over_cap") != 0):
            raise ValueError("Variable identifier audit coverage differs")
        if part.get("target_tokens_min_max") != [target_tokens, target_tokens]:
            raise ValueError("Every target must contain exactly L plus EOS tokens")
        lengths = part.get("input_tokens_p50_p90_p99_max")
        if not isinstance(lengths, list) or len(lengths) != 4 or lengths[-1] > 512:
            raise ValueError("Every input must remain within the 512-token limit")
        if part.get("atomic_identifier_tokens") is not True:
            raise ValueError("Every identifier token must remain atomic")
    return report


def expand_items(codes: list[str], catalog: dict[str, list[int]], limit: int = 10) -> list[int]:
    expanded = []
    seen = set()
    for code in codes:
        for item in sorted(catalog[code]):
            if item not in seen:
                seen.add(item)
                expanded.append(item)
                if len(expanded) == limit:
                    return expanded
    return expanded


def validate_evaluation_rows(
    rows: list[dict], expected_users: list[str], catalog: dict[str, list[int]],
    beams: int, max_new_tokens: int, bank: dict,
) -> dict:
    if beams != 20 or max_new_tokens < bank["target_tokens"]:
        raise ValueError("Figure8 generation budget cannot represent the identifier")
    users = [row.get("user") for row in rows]
    if users != expected_users or len(set(users)) != len(expected_users):
        raise ValueError("Figure8 evaluation coverage differs from the all-user order")
    item_predictions = {}
    for row in rows:
        codes = row.get("codes")
        if not isinstance(codes, list) or len(codes) != beams or len(set(codes)) != beams:
            raise ValueError("Figure8 evaluation requires 20 complete unique beams")
        if any(code not in catalog for code in codes):
            raise ValueError("Figure8 evaluation emitted a code outside the catalog")
        item_predictions[row["user"]] = expand_items(codes, catalog)
    return {
        "users": len(users),
        "beams": beams,
        "max_new_tokens": max_new_tokens,
        "item_predictions": item_predictions,
    }


def bind_tokenizer_queue(work: Path, binding: dict) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    marker = work / "figure8-tokenizer-binding.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != binding:
        raise ValueError("Figure8 tokenizer queue is frozen to different bytes")
    if not marker.exists():
        write_json(marker, binding)
    return marker


def queue_binding(
    *, plan_sha256: str, work: Path, durable_root: Path,
    gpu_indexes: tuple[int, int], source_sha256: dict,
) -> dict:
    return {
        "kind": "figure8_tokenizer_queue_v1",
        "plan_sha256": plan_sha256,
        "work": str(work),
        "durable_root": str(durable_root),
        "gpu_indexes": list(gpu_indexes),
        "queue": tokenizer_cells(),
        "selection": "within-arm validation only; no cross-arm or test selection",
        "source_sha256": source_sha256,
        "admission_policy":
            "individual immutable arm receipts may arrive after queue binding",
    }


def verify_durable_tree(path: Path) -> dict:
    marker = path / "bank-receipt.json"
    if path.is_symlink() or not marker.is_file() or marker.is_symlink():
        raise ValueError("Figure8 arm durable receipt is missing or redirected")
    receipt = json.loads(marker.read_text(encoding="utf-8"))
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Figure8 arm durable receipt has no inventory")
    actual = {
        candidate.relative_to(path).as_posix()
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate != marker
    }
    if actual != set(files):
        raise ValueError("Figure8 arm durable inventory differs")
    for name, metadata in files.items():
        target = path / name
        if (target.is_symlink() or target.stat().st_size != metadata.get("bytes")
                or file_sha256(target) != metadata.get("sha256")):
            raise ValueError(f"Figure8 arm durable checksum differs: {name}")
    return {"bank_receipt_sha256": file_sha256(marker), "files": len(files)}


def require_durable_admission_path(
    path: Path, destination: Path, arm: str, work_basename: str,
) -> Path:
    expected = (
        destination / "openllama-v2" / work_basename
        / "operational" / "admissions" / arm / "admission.json"
    )
    if (path.absolute() != expected.absolute()
            or path.resolve() != expected.resolve()
            or path.is_symlink()):
        raise ValueError("Figure8 arm must use its exact durable admission namespace")
    return path.resolve()


def verify_arm_compatibility(arm: str, audit: dict, payload: Path) -> dict | None:
    if arm not in KMEANS_COMPATIBILITY_ARMS:
        return None
    admission_sha256 = audit.get("kmeans_compatibility_admission_sha256")
    if not isinstance(admission_sha256, str) or not re.fullmatch(
        r"[a-f0-9]{64}", admission_sha256,
    ):
        raise ValueError("A trusted tokenizer compatibility admission digest is required")
    from figure8_tokenizer_recovery import verify_completed_compatibility
    return verify_completed_compatibility(payload, arm, admission_sha256)


def verify_arm_admission(
    arm: str, audit: dict, spec_path: Path, durable_root: Path,
    expected_spec: dict, execution: dict,
) -> dict:
    if arm not in ARM_ORDER or audit.get("status") != "pass" or audit.get("arm") != arm:
        raise ValueError("Figure8 arm audit identity differs")
    bundle_name = audit.get("bundle", "")
    if not re.fullmatch(re.escape(arm) + r"-completed-[A-Za-z0-9_.-]+", bundle_name):
        raise ValueError("Figure8 arm bundle name is not bound to its arm")
    if spec_path.is_symlink():
        raise ValueError("Figure8 arm specification is redirected")
    launch_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    validate_arm_launch_spec(launch_spec, expected_spec, arm)
    bundle = durable_root / bundle_name
    durable = verify_durable_tree(bundle)
    payload = bundle / "payload"
    compatibility = verify_arm_compatibility(arm, audit, payload)
    config = json.loads((payload / "run-config.json").read_text(encoding="utf-8"))
    codes = np.load(payload / "export" / "codes.npy", allow_pickle=False)
    indices = json.loads((payload / "export" / "index.json").read_text(encoding="utf-8"))
    validate_arm_run_config(config, expected_spec, execution, len(indices))
    bank = validate_identifier_bank(
        indices, codes, len(indices), list(expected_spec["codes"]),
    )
    required = {
        "bundle_completed_sha256": bundle / "completed.json",
        "run_completed_sha256": payload / "completed.json",
        "index_sha256": payload / "export" / "index.json",
        "codes_sha256": payload / "export" / "codes.npy",
    }
    if (audit.get("final_epoch") != 20000
            or type(audit.get("selected_epoch")) is not int
            or not 1 <= audit["selected_epoch"] <= 20000
            or durable["bank_receipt_sha256"] != audit.get("durable_bank_receipt_sha256")
            or any(file_sha256(path) != audit.get(field) for field, path in required.items())):
        raise ValueError("Figure8 arm final or durable checksum identity differs")
    result = {
        "arm": arm,
        "bundle": bundle_name,
        "selected_epoch": audit["selected_epoch"],
        "final_epoch": audit["final_epoch"],
        "spec_sha256": file_sha256(spec_path),
        "bank_receipt_sha256": durable["bank_receipt_sha256"],
        "bank": bank,
        **{field: audit[field] for field in required},
    }
    if compatibility is not None:
        result["kmeans_compatibility"] = compatibility
    return result


def derive_arm_manifest(
    parent: dict, parent_sha256: str, lineage: dict, indices: dict,
    codes: np.ndarray, codebook_sizes: list[int],
) -> dict:
    dataset = "Instruments"
    source = parent["datasets"][dataset]
    keys = (
        "items", "users", "interactions", "training_prefix_examples",
        "validation_users", "test_users", "sha256",
    )
    reference = {key: copy.deepcopy(source[key]) for key in keys}
    bank = validate_identifier_bank(
        indices, codes, reference["items"], codebook_sizes,
    )
    reference["sha256"]["index"] = lineage["index_sha256"]
    reference["figure8_identifier_bank"] = bank
    return {
        "schema_version": 1,
        "kind": "figure8_tokenizer_downstream_inputs_v1",
        "status": "derived_inputs_not_training_results",
        "parent_manifest_sha256": parent_sha256,
        "experiment_lineage": copy.deepcopy(lineage),
        "datasets": {dataset: reference},
        "claim":
            "One-factor Figure8 tokenizer bank with fresh random native T5; "
            "compatible-backbone method replication.",
    }


def validate_stage_lineage(manifest: dict, expected: dict) -> dict:
    if manifest.get("experiment_lineage") != expected:
        raise ValueError("cross-arm or stale Figure8 stage lineage")
    return manifest


def validate_resume_identity(run: dict, cell: dict, bundle_completed_sha256: str) -> dict:
    expected = {
        "arm": cell["arm"],
        "run_id": cell["run_id"],
        "stage_id": cell["stage_id"],
        "bundle_completed_sha256": bundle_completed_sha256,
        "seed": 42,
        "temperature": 0.7,
    }
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError("cross-arm or stale Figure8 resume identity")
    if run.get("initialization") != "author random T5 configuration":
        raise ValueError("Figure8 arms may not warm-start from another model")
    return run


def completed_evaluation(root: Path) -> Path | None:
    completed = [
        path for path in sorted(root.glob("evaluation-attempt-*"))
        if (path / "completed.json").is_file()
    ]
    if len(completed) > 1:
        raise ValueError("Ambiguous completed evaluations; never select by test performance")
    return completed[0] if completed else None


def execute_cell(runtime, cell: dict, admission: dict) -> dict:
    stage = runtime.admit(cell, admission)
    runtime.gate(cell, stage)
    run = runtime.train(cell, stage)
    runtime.bank_run(cell, run)
    evaluation = runtime.evaluate(cell, stage, run)
    runtime.bank_evaluation(cell, evaluation)
    return runtime.publish(cell, stage, run, evaluation)


def next_admitted_cell(
    cells: list[dict], admissions: dict[str, dict], banked_run_ids: set[str],
) -> dict | None:
    for cell in cells:
        if cell["run_id"] not in banked_run_ids and cell["arm"] in admissions:
            return cell
    return None


def aggregate_panels(points: dict[str, dict], baseline: dict) -> dict:
    if set(points) != set(ARM_ORDER):
        raise ValueError("Figure8 panel aggregation requires all 15 nonbaseline arms")
    panels = {}
    for panel in PANEL_ORDER:
        arms = [arm for arm in ARM_ORDER if arm.startswith(panel + "-")]
        panels[panel] = {
            "baseline": baseline,
            "points": [{"arm": arm, **points[arm]} for arm in arms],
        }
    result = {
        "status": "pass",
        "kind": "figure8_tokenizer_panels_v1",
        "dataset": "Instruments",
        "fixed_temperature": 0.7,
        "panels": panels,
        "selection": "descriptive only; no test-based winner",
    }
    if contains_winner_field(result):
        raise ValueError("Figure8 panel result may not contain a test-derived winner")
    return result
