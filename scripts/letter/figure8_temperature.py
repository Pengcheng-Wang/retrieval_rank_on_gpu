"""Separate, frozen Figure 8 temperature queue over the baseline full tokenizer."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from fresh_ids import file_sha256, read_json, write_json
from fresh_pipeline import Runtime, cell_is_banked


TEMPERATURES = (("0p6", 0.6), ("0p8", 0.8), ("0p9", 0.9), ("1p1", 1.1), ("1p2", 1.2))
TOKENIZER_ARMS = (
    "alpha-0p001", "alpha-0p01", "alpha-0p1",
    "beta-0p00001", "beta-0p001", "beta-0p01",
    "clusters-5", "clusters-15", "clusters-20",
    "levels-2", "levels-6", "levels-8",
    "codes-64", "codes-128", "codes-512",
)


def temperature_cells() -> list[dict]:
    return [
        {
            "dataset": "Instruments", "variant": "full", "seed": 42,
            "temperature": temperature,
            "rq_run_id": "openllama-v2-rq-Instruments-full-s42",
            "stage_id": "Instruments-full-ids-v1",
            "run_id": f"openllama-v2-figure8-t5-Instruments-full-s42-t{label}-v1",
        }
        for label, temperature in TEMPERATURES
    ]


def parse_gpu_pair(value: str) -> tuple[int, int]:
    if not re.fullmatch(r"\d+,\d+", value):
        raise ValueError("GPU pair must contain two comma-separated physical indexes")
    indexes = tuple(map(int, value.split(",")))
    if len(set(indexes)) != 2:
        raise ValueError("GPU pair must contain two distinct physical indexes")
    return indexes


def bind_temperature_queue(work: Path, binding: dict) -> Path:
    marker = work / "figure8-temperature-binding.json"
    if marker.exists() and read_json(marker) != binding:
        raise ValueError("Figure8 temperature queue is frozen to different bytes")
    if not marker.exists():
        write_json(marker, binding)
    return marker


def execute_next(runtime: Runtime, bundle: Path, cells: list[dict]) -> dict | None:
    pending = [cell for cell in cells if not cell_is_banked(runtime.work, cell)]
    if not pending:
        return None
    cell = pending[0]
    runtime.execute_cell(cell, bundle)
    return cell


def load_plan(path: Path) -> dict:
    plan = read_json(path)
    expected = {
        "kind": "letter_figure8_temperature_v1",
        "dataset": "Instruments",
        "variant": "full",
        "seed": 42,
        "temperatures": [temperature for _, temperature in TEMPERATURES],
        "baseline_temperatures": [0.7, 1.0],
        "learning_rate": 0.0005,
        "epochs": 200,
        "world_size": 2,
        "effective_batch_size": 1024,
        "beams": 20,
        "bootstrap_seed": 20261027,
        "bootstrap_replicates": 2000,
        "seen_item_exclusion": False,
        "selection":
            "within-arm minimum validation loss with patience20; never compare validation loss across temperatures",
        "comparison_reference": 1.0,
        "comparison_metrics": ["recall@5", "recall@10", "ndcg@5", "ndcg@10"],
        "metric_identities": ["code", "item"],
        "test_selection": "descriptive only; no best-temperature field or recipe selection",
        "tokenizer_barrier": list(TOKENIZER_ARMS),
        "baseline_rq_run_id": "openllama-v2-rq-Instruments-full-s42",
        "baseline": {
            "bundle_name":
                "openllama-v2-rq-Instruments-full-s42-completed-8a999c191f8ec7fd",
            "bundle_completed_sha256":
                "621fff5a7e7f42e425b3ca45b4f61fcd23c808d439a1054002a842422cd6579e",
            "run_completed_sha256":
                "c88a19484d073ab5e119c0e6d4062014c75c8755055c0b9bb885e1ba40d03af9",
            "final_state_sha256":
                "796d484fab903b239e02e2c90a2323073408e6f875ba9178c76cb78329ded771",
            "selected_epoch": 18000,
            "selected_state_sha256":
                "59a2dcc4f61c6c9fd15d96d946db33fa590a9689af55e7b08400d48eb2d001f9",
            "index_sha256":
                "44048e87910d0367ebe1733e1fcec6d833f2d51b02241fc3edc5800194e82a74",
            "codes_sha256":
                "00444111176fd11997ab4c6450dffd417360dec501506b0a9133fb104a2c2688",
            "durable_bank_receipt_sha256":
                "8a6e3ebbc42bd602ab7c114b54598f10f35d946280cedb01e9bf7b3cd064f08f",
        },
        "baseline_results": {
            "temperature_0p7_result_sha256":
                "2c7cfc6ba34a28f00cd8f154b3de6686124e60277841dc0736886afd78ba61a8",
            "temperature_1p0_result_sha256":
                "b76344c10082fbe6bdd3a0ac11390942bee387e4edf38e4ed9e36478de1abafe",
        },
        "reference_fresh_queue_cells": 39,
        "work_basename": "fresh-t5-figure8-temperature-v1",
        "durability":
            "separate node0-only work/durable namespace under the common bank lock",
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("Figure8 temperature plan differs from the frozen scientific grid")
    return plan


def validate_admission(path: Path, expected_sha256: str, plan: dict, bundle: Path) -> dict:
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise ValueError("Figure8 admission requires an external digest")
    if file_sha256(path) != expected_sha256:
        raise ValueError("Figure8 admission is not externally authorized")
    admission = read_json(path)
    tokenizer_arms = admission.get("tokenizer_arms")
    barriers_valid = (
        isinstance(tokenizer_arms, dict)
        and set(tokenizer_arms) == set(TOKENIZER_ARMS)
        and all(
            value.get("status") == "verified_durable_final"
            and value.get("final_epoch") == 20000
            and re.fullmatch(r"[a-f0-9]{64}", value.get("bundle_completed_sha256", ""))
            and re.fullmatch(r"[a-f0-9]{64}", value.get("bank_receipt_sha256", ""))
            for value in tokenizer_arms.values()
        )
    )
    if (admission.get("status") != "pass"
            or admission.get("kind") != "figure8_temperature_admission_v1"
            or admission.get("temperatures") != plan["temperatures"]
            or admission.get("baseline_bundle") != bundle.name
            or admission.get("baseline_bundle_completed_sha256")
            != file_sha256(bundle / "completed.json")
            or admission.get("baseline") != plan["baseline"]
            or admission.get("all_tokenizer_one_factor_arms_durable") is not True
            or not barriers_valid
            or admission.get("existing_fresh_queue_unchanged") is not True
            or admission.get("existing_fresh_queue_cells") != 39
            or not re.fullmatch(
                r"[a-f0-9]{64}", admission.get("existing_fresh_queue_binding_sha256", ""),
            )):
        raise ValueError("Figure8 temperature prerequisites are incomplete or changed")
    return admission


def queue_binding(
    runtime: Runtime, plan_path: Path, admission_path: Path,
    bundle: Path, reference_binding: Path,
) -> dict:
    return {
        "kind": "figure8_temperature_queue_v1",
        "plan_sha256": file_sha256(plan_path),
        "admission_sha256": file_sha256(admission_path),
        "baseline_bundle": bundle.name,
        "baseline_bundle_completed_sha256": file_sha256(bundle / "completed.json"),
        "reference_fresh_queue_binding_sha256": file_sha256(reference_binding),
        "work": str(runtime.work),
        "durable_root": str(runtime.bank_root),
        "control": str(runtime.control),
        "gpu_indexes": list(runtime.gpu_indexes),
        "cells": temperature_cells(),
        "selection": "within-arm minimum validation loss only; no cross-temperature selection",
        "comparison": "all six temperatures versus predeclared temperature1 reference; test descriptive only",
    }


def driver(
    runtime: Runtime, bundle: Path, plan_path: Path, admission_path: Path,
    admission_sha256: str, reference_binding: Path, checks: int,
) -> None:
    import fcntl

    plan = load_plan(plan_path)
    admission = validate_admission(admission_path, admission_sha256, plan, bundle)
    if runtime.work.name != plan["work_basename"]:
        raise ValueError("Figure8 temperature work namespace differs")
    if (admission.get("gpu_indexes") != list(runtime.gpu_indexes)
            or admission.get("fresh_queue_gpu_indexes") != [2, 3]
            or set(admission["gpu_indexes"]) & set(admission["fresh_queue_gpu_indexes"])
            or admission.get("same_allocation_distinct_gpu_lease") is not True):
        raise ValueError("Figure8 and fresh queues do not have distinct declared GPU leases")
    binding = queue_binding(runtime, plan_path, admission_path, bundle, reference_binding)
    with (runtime.work / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = bind_temperature_queue(runtime.work, binding)
        write_json(runtime.work / "driver.json", {
            "pid": os.getpid(), "started_unix": time.time(),
            "binding_sha256": file_sha256(marker), "checks": checks,
        })
        for check in range(checks):
            cell = execute_next(runtime, bundle, binding["cells"])
            if cell is None:
                write_json(runtime.work / "status.json", {
                    "status": "all_temperature_arms_durably_complete",
                })
                return
            write_json(runtime.work / "status.json", {
                "status": "completed_one_arm", "check": check + 1, "cell": cell,
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "binding", "project", "control", "work", "bundle", "plan",
        "admission", "reference-binding",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--gpu-pair", required=True)
    parser.add_argument("--checks", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.checks <= len(TEMPERATURES):
        parser.error(f"Use 1..{len(TEMPERATURES)} checks")
    gpu_indexes = parse_gpu_pair(args.gpu_pair)
    runtime = Runtime(
        args.binding, args.project, args.control, args.work,
        gpu_indexes=gpu_indexes,
    )
    driver(
        runtime, args.bundle.resolve(), args.plan.resolve(), args.admission.resolve(),
        args.admission_sha256, args.reference_binding.resolve(), args.checks,
    )


if __name__ == "__main__":
    main()
