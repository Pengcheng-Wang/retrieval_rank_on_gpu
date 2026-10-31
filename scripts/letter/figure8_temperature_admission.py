"""Create the immutable admission receipt for the separate Figure 8 temperature queue."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from bank import copy_artifact
from figure8_temperature import (
    TOKENIZER_ARMS,
    SWEEP_TEMPERATURES,
    assemble_preflight,
    control_source_hashes,
    load_plan,
    parse_gpu_pair,
    validate_admission,
    validate_preflight,
    verify_baseline,
    verify_durable_tree,
    verify_fresh_queue_artifacts,
    verify_tokenizer_barrier,
)
from fresh_ids import (
    file_sha256,
    load_execution,
    read_json,
    validate_scope,
    verify_durable,
    verify_rq,
    write_json,
)
from fresh_pipeline import queue_cells
from fresh_rq_admission import require_unredirected


def parse_audits(values: list[str]) -> dict[str, Path]:
    audits = {}
    for value in values:
        arm, separator, path = value.partition("=")
        if not separator or arm not in TOKENIZER_ARMS or arm in audits:
            raise ValueError("Use each exact --tokenizer-audit ARM=PATH once")
        audits[arm] = Path(path)
    if set(audits) != set(TOKENIZER_ARMS):
        raise ValueError("All 15 tokenizer final audits are required")
    return audits


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "binding", "project", "control", "fresh-work", "temperature-work",
        "bundle", "plan", "preflight", "reference-binding", "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--gpu-pair", required=True)
    parser.add_argument("--tokenizer-audit", action="append", default=[])
    args = parser.parse_args()
    import fcntl

    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise RuntimeError("Figure8 admission is CPU-only")
    root, destination = validate_scope(args.binding)
    project = require_unredirected(args.project, root, "Figure8 project")
    control = require_unredirected(args.control, root, "Figure8 control")
    fresh_work = require_unredirected(args.fresh_work, root, "fresh queue work")
    temperature_work = require_unredirected(
        args.temperature_work, root, "temperature work",
    )
    plan_path = require_unredirected(args.plan, control, "temperature plan")
    bundle = require_unredirected(args.bundle, root, "baseline bundle")
    preflight_path = require_unredirected(
        args.preflight, temperature_work, "temperature preflight",
    )
    reference_binding = require_unredirected(
        args.reference_binding, fresh_work, "fresh queue binding",
    )
    output_dir = require_unredirected(
        args.output_dir, temperature_work, "temperature admission output",
    )
    if (project != root / "project"
            or not control.is_relative_to(root)
            or fresh_work != root / "fresh-t5-v3"
            or temperature_work != root / "fresh-t5-figure8-temperature-v1"
            or not bundle.is_relative_to(root)
            or preflight_path != temperature_work / "temperature-preflight-v1" / "receipt.json"
            or reference_binding.parent != fresh_work
            or output_dir.parent != temperature_work
            or output_dir.name != "temperature-admission-v1"):
        raise ValueError("Figure8 admission paths escape the separate owned namespaces")
    plan = load_plan(plan_path)
    gpu_indexes = parse_gpu_pair(args.gpu_pair)
    if (set(gpu_indexes) & {2, 3}) or plan["work_basename"] != temperature_work.name:
        raise ValueError("Figure8 admission requires a lease distinct from fresh GPU2/3")
    execution = control / "configs" / "letter-fresh-id-execution-v1.json"
    manifest_path = control / "configs" / "letter-reproduction-v1.json"
    method = control / "configs" / "letter-openllama-v2-method-v1.json"
    tokenizer_recipe = control / "configs" / "letter-tokenizer-reconstruction-v1.json"
    spec = load_execution(execution, manifest_path, method, tokenizer_recipe)
    manifest = read_json(manifest_path)
    expected_queue = queue_cells(spec)
    audits = {
        arm: require_unredirected(path, root, f"{arm} final audit")
        for arm, path in parse_audits(args.tokenizer_audit).items()
    }
    driver_lock = fresh_work / "driver.lock"
    with driver_lock.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("The existing 39-cell fresh queue driver is still active") from error
        with (root / "bank.lock").open("a") as bank_lock:
            fcntl.flock(bank_lock, fcntl.LOCK_SH)
            queue_acks = verify_fresh_queue_artifacts(
                fresh_work,
                reference_binding,
                expected_queue,
                destination / "openllama-v2" / "fresh-t5-v3",
            )
            rq = verify_rq(bundle, spec, manifest, "Instruments", "full")
            durable = verify_durable(
                bundle, destination / "openllama-v2" / "tokenizers" / bundle.name,
            )
            baseline_results = {
                "temperature_0p7_result_sha256":
                    fresh_work / "results"
                    / "openllama-v2-fresh-t5-Instruments-full-s42-t07-v1"
                    / "result.json",
                "temperature_1p0_result_sha256":
                    fresh_work / "results"
                    / "openllama-v2-fresh-t5-Instruments-full-s42-t10-v1"
                    / "result.json",
            }
            baseline = verify_baseline(
                plan, bundle, rq, durable, baseline_results,
            )
            tokenizer_arms = verify_tokenizer_barrier(
                audits,
                destination / "openllama-v2" / "figure8-tokenizers",
                root / "figure8-sweeps-v1" / "specs",
            )
            preflight = validate_preflight(read_json(preflight_path))
            recomputed_preflight = assemble_preflight(
                temperature_work / "temperature-preflight-v1" / "probes",
                temperature_work / "stages" / "Instruments-full-ids-v1" / "stage.json",
                temperature_work / "audits" / "Instruments-full-ids-v1" / "audit.json",
                plan["preflight_batches_per_rank"],
            )
            preflight_durable = verify_durable_tree(
                destination / "openllama-v2" / temperature_work.name
                / "operational" / "temperature-preflight-v1",
            )
            if (file_sha256(preflight_path)
                    != file_sha256(
                        destination / "openllama-v2" / temperature_work.name
                        / "operational" / "temperature-preflight-v1" / "receipt.json"
                    )
                    or preflight.get("plan_sha256") != file_sha256(plan_path)
                    or preflight.get("temperatures") != plan["preflight_temperatures"]
                    or preflight.get("batches_per_rank")
                    != plan["preflight_batches_per_rank"]
                    or preflight.get("gpu_indexes") != list(gpu_indexes)
                    or preflight.get("probe_source_sha256")
                    != file_sha256(control / "scripts/letter/figure8_temperature_probe.py")
                    or preflight.get("preflight_source_sha256")
                    != file_sha256(control / "scripts/letter/figure8_temperature_preflight.py")
                    or any(preflight.get(key) != value
                           for key, value in recomputed_preflight.items())):
                raise ValueError("Figure8 temperature preflight is stale or undurable")
    if output_dir.exists():
        raise FileExistsError("Figure8 temperature admission is immutable")
    output_dir.mkdir()
    admission = {
        "status": "pass",
        "kind": "figure8_temperature_admission_v1",
        "temperatures": plan["temperatures"],
        "baseline_bundle": bundle.name,
        "baseline_bundle_completed_sha256": file_sha256(bundle / "completed.json"),
        "baseline": baseline,
        "baseline_results": plan["baseline_results"],
        "all_tokenizer_one_factor_arms_durable": True,
        "tokenizer_arms": tokenizer_arms,
        "existing_fresh_queue_unchanged": True,
        "existing_fresh_queue_cells": len(expected_queue),
        "existing_fresh_queue_binding_sha256": file_sha256(reference_binding),
        "fresh_queue_durably_complete": True,
        "fresh_queue_driver_inactive": True,
        "fresh_queue_result_acks": queue_acks,
        "source_sha256": control_source_hashes(control / "scripts" / "letter"),
        "temperature_preflight": {
            "receipt_sha256": file_sha256(preflight_path),
            "durable_bank_receipt_sha256":
                preflight_durable["bank_receipt_sha256"],
            "stage_sha256": preflight["stage_sha256"],
            "audit_sha256": preflight["audit_sha256"],
            "initial_model_identical": True,
            "ordered_batches_identical": True,
            "temperatures": plan["preflight_temperatures"],
            "batches_per_rank": plan["preflight_batches_per_rank"],
        },
        "gpu_indexes": list(gpu_indexes),
        "fresh_queue_gpu_indexes": [2, 3],
        "same_allocation_distinct_gpu_lease": True,
        "claim":
            "Admission only; no test metric was read to select a tokenizer or temperature.",
    }
    write_json(output_dir / "admission.json", admission)
    (output_dir / "README.md").write_text(
        "# Figure 8 temperature admission v1\n\n"
        "This receipt proves the 15 tokenizer barrier, exact full-tokenizer baseline, "
        "completed inactive 39-cell queue, distinct GPU lease, and identical "
        "initialization/order preflight without selecting on test outcomes.\n",
        encoding="utf-8",
    )
    admission_sha256 = file_sha256(output_dir / "admission.json")
    validate_admission(output_dir / "admission.json", admission_sha256, plan, bundle)
    target = (
        destination / "openllama-v2" / temperature_work.name
        / "operational" / output_dir.name
    )
    with (root / "bank.lock").open("a") as bank_lock:
        fcntl.flock(bank_lock, fcntl.LOCK_EX)
        copy_artifact(output_dir, target)
    print(json.dumps({
        "status": "pass",
        "admission_sha256": admission_sha256,
        "durable": str(target),
    }), flush=True)


if __name__ == "__main__":
    main()
